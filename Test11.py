import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 51 # Neuer Seed
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"INFO: Globaler Seed wurde auf {seed_value} gesetzt.")

# --- GPU-Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP_GLOBAL = torch.cuda.is_available()
print(f"Verwende Gerät: {DEVICE}, AMP global aktiviert: {USE_AMP_GLOBAL}")

# --- DPPLayer_SharedG Klasse (unverändert) ---
class DPPLayer_SharedG(nn.Module):
    def __init__(self, input_features, output_features, shared_g_dim):
        super(DPPLayer_SharedG, self).__init__()
        self.input_features = input_features; self.output_features = output_features
        self.shared_g_dim = shared_g_dim
        self.w_a = nn.Parameter(torch.Tensor(output_features, input_features))
        self.b_a = nn.Parameter(torch.Tensor(output_features))
        self.w_b = nn.Parameter(torch.Tensor(output_features, input_features))
        self.b_b = nn.Parameter(torch.Tensor(output_features))
        self.w_g_shared = nn.Parameter(torch.Tensor(shared_g_dim, input_features))
        self.b_g_shared = nn.Parameter(torch.Tensor(shared_g_dim))
        self.w_g_unit = nn.Parameter(torch.Tensor(output_features, shared_g_dim))
        self.b_g_unit = nn.Parameter(torch.Tensor(output_features))
        self._init_weights()

    def _init_weights(self):
        for w, b in [(self.w_a, self.b_a), (self.w_b, self.b_b)]:
            nn.init.kaiming_uniform_(w, a=(5**0.5))
            if b is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w); bound = 1/(fan_in**0.5) if fan_in>0 else 0; nn.init.uniform_(b,-bound,bound)
        nn.init.kaiming_uniform_(self.w_g_shared, a=(5**0.5))
        if self.b_g_shared is not None:
            fan_in_g_shared,_ = nn.init._calculate_fan_in_and_fan_out(self.w_g_shared); bound_g_shared = 1/(fan_in_g_shared**0.5) if fan_in_g_shared>0 else 0; nn.init.uniform_(self.b_g_shared, -bound_g_shared, bound_g_shared)
        nn.init.kaiming_uniform_(self.w_g_unit, a=(5**0.5))
        if self.b_g_unit is not None:
            fan_in_g_unit,_ = nn.init._calculate_fan_in_and_fan_out(self.w_g_unit); bound_g_unit = 1/(fan_in_g_unit**0.5) if fan_in_g_unit > 0 else 0; nn.init.uniform_(self.b_g_unit, -bound_g_unit, bound_g_unit)

    def forward(self, x, return_alpha=False):
        z_a=F.linear(x,self.w_a,self.b_a); z_b=F.linear(x,self.w_b,self.b_b)
        x_shared_g=F.linear(x,self.w_g_shared,self.b_g_shared)
        g_logits=F.linear(x_shared_g,self.w_g_unit,self.b_g_unit)
        alpha=torch.sigmoid(g_logits); z_final=alpha*z_a+(1-alpha)*z_b
        if return_alpha: return z_final,alpha
        return z_final

# --- Modell-Definitionen (DPPModelBase bleibt gleich) ---
class DPPModelBase(nn.Module):
    def __init__(self, dpp_layer, hidden_size_dpp, output_size):
        super(DPPModelBase, self).__init__(); self.dpp_layer1 = dpp_layer
        self.relu1 = nn.ReLU(); self.fc_out = nn.Linear(hidden_size_dpp, output_size)
        self.last_alphas = None

    def forward(self, x, return_alpha_flag=False):
        if return_alpha_flag and hasattr(self.dpp_layer1, 'forward') and \
           'return_alpha' in self.dpp_layer1.forward.__code__.co_varnames:
            out, alphas = self.dpp_layer1(x, return_alpha=True)
            self.last_alphas = alphas
        else:
            out = self.dpp_layer1(x, return_alpha=False)
        out = self.relu1(out); out = self.fc_out(out)
        return out

# --- Datengenerierung für "Einfacher Instruktions-Interpreter" ---
class InstructionInterpreterDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, num_instructions=8):
        super(InstructionInterpreterDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions # LOAD_R0_X, LOAD_R1_X, XOR, AND, NOT_R0, OUT_R0, OUT_R1, NO_OP

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            # Register und vorheriger Output
            r0 = 0.0
            r1 = 0.0
            previous_y = 0.0

            for t in range(self.seq_len):
                current_x_bit = float(np.random.randint(0, 2))
                instr_idx = np.random.randint(0, self.num_instructions)
                instruction_one_hot = self._to_one_hot(instr_idx, self.num_instructions)

                # Inputs für das Modell (mit Rauschen)
                x_t_noisy = current_x_bit + np.random.normal(0, self.noise_level)
                # Rauschen zu One-Hot ist knifflig, wir lassen es für den One-Hot-Teil weg,
                # oder addieren es und clippen/normalisieren es. Für Einfachheit: kein Rauschen auf One-Hot.
                # instruction_one_hot_noisy = instruction_one_hot + np.random.normal(0, self.noise_level/5, size=instruction_one_hot.shape)
                # instruction_one_hot_noisy = np.clip(instruction_one_hot_noisy, 0, 1) # Beispielhafte Behandlung

                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level) # Vorheriger Zustand
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level) # Vorheriger Zustand
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)


                input_features = [x_t_noisy] + list(instruction_one_hot) + \
                                 [r0_tm1_noisy, r1_tm1_noisy, y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Zielberechnung y_t und Register-Updates (basierend auf sauberen Werten)
                # Die Register R0, R1 werden hier für den *nächsten* Schritt aktualisiert
                # Der Output y_t basiert auf den Registern *nach* der aktuellen Operation, wenn es eine OUT-Inst ist.
                
                # Temporäre Register für die Berechnung des aktuellen Schritts
                current_r0 = r0
                current_r1 = r1
                target_y = 0.0 # Default für nicht-Output Instruktionen

                if instr_idx == 0: # LOAD_R0_X
                    current_r0 = current_x_bit
                elif instr_idx == 1: # LOAD_R1_X
                    current_r1 = current_x_bit
                elif instr_idx == 2: # XOR_R0_R1 (Store in R0)
                    current_r0 = float(int(r0) ^ int(r1))
                elif instr_idx == 3: # AND_R0_R1 (Store in R0)
                    current_r0 = float(int(r0) & int(r1))
                elif instr_idx == 4: # NOT_R0 (Store in R0)
                    current_r0 = float(1 - int(r0))
                elif instr_idx == 5: # OUT_R0
                    target_y = current_r0 # R0 *dieses* Schrittes (kann durch obige Ops verändert worden sein)
                elif instr_idx == 6: # OUT_R1
                    target_y = current_r1 # R1 *dieses* Schrittes
                elif instr_idx == 7: # NO_OP
                    target_y = previous_y # Behalte vorherigen Output

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                # Update der "echten" Register für den nächsten Input-Zeitschritt
                r0 = current_r0
                r1 = current_r1
                previous_y = target_y # Der tatsächliche y_t wird zum y_{t-1} des nächsten Schritts

    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen mit AMP (unverändert) ---
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None, scheduler_obj=None):
    model.train()
    history={'loss':[],'accuracy':[],'time_per_epoch':[]}
    time_to_target_accuracy=None; epoch_at_target_accuracy=None
    total_training_start_time=time.time()
    for epoch in range(epochs):
        epoch_start_time=time.time()
        epoch_loss=0; correct_preds=0; total_preds=0; num_batches = 0
        for batch_idx, (batch_inputs,batch_labels) in enumerate(train_loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch: break
            num_batches += 1
            batch_inputs,batch_labels=batch_inputs.to(DEVICE),batch_labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type,enabled=USE_AMP_GLOBAL):
                outputs=model(batch_inputs); loss=criterion(outputs,batch_labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss+=loss.item()
            with torch.no_grad(): preds=torch.sigmoid(outputs)>0.5
            correct_preds+=(preds==batch_labels.bool()).sum().item(); total_preds+=batch_labels.size(0)
        
        time_this_epoch = time.time() - epoch_start_time
        avg_epoch_loss=epoch_loss/num_batches if num_batches > 0 else 0
        accuracy=correct_preds/total_preds if total_preds > 0 else 0
        history['loss'].append(avg_epoch_loss); history['accuracy'].append(accuracy); history['time_per_epoch'].append(time_this_epoch)

        if scheduler_obj:
            if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_obj.step(accuracy) # Basiere auf Validierungs-Acc, hier nehmen wir Train-Acc für Einfachheit
            else:
                scheduler_obj.step() # Für CosineAnnealing etc. pro Epoche

        if (accuracy>=target_accuracy and epoch_at_target_accuracy is None) or (epoch+1)%(epochs//10 if epochs>=10 else 1)==0 or epoch==0 or epoch==epochs-1:
             current_total_time=time.time()-total_training_start_time
             lr_print = optimizer.param_groups[0]['lr']
             print(f"Epoch [{epoch+1}/{epochs}] {model_name}, Loss: {avg_epoch_loss:.4f}, Acc: {accuracy:.4f}, LR: {lr_print:.1e}, Total Time: {current_total_time:.2f}s")
        if accuracy>=target_accuracy and epoch_at_target_accuracy is None:
            time_to_target_accuracy=time.time()-total_training_start_time; epoch_at_target_accuracy=epoch+1
            print(f"--- {model_name} reached target accuracy of {target_accuracy*100:.1f}% at epoch {epoch_at_target_accuracy} in {time_to_target_accuracy:.3f}s ---")
    total_training_time=time.time()-total_training_start_time
    print(f"--- Training {model_name} finished in {total_training_time:.3f}s ---")
    return history,total_training_time,epoch_at_target_accuracy,time_to_target_accuracy

def evaluate_model_amp(model,data_loader,criterion,model_name="Model", steps_per_epoch=None):
    model.eval(); total_loss=0; correct_preds=0; total_preds=0; num_batches = 0
    with torch.no_grad():
        for batch_idx, (batch_inputs,batch_labels) in enumerate(data_loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch: break
            num_batches +=1
            batch_inputs,batch_labels=batch_inputs.to(DEVICE),batch_labels.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type,enabled=USE_AMP_GLOBAL):
                outputs=model(batch_inputs); loss=criterion(outputs,batch_labels)
            total_loss+=loss.item()
            preds=torch.sigmoid(outputs)>0.5
            correct_preds+=(preds==batch_labels.bool()).sum().item(); total_preds+=batch_labels.size(0)
    avg_loss=total_loss/num_batches if num_batches > 0 else 0
    accuracy=correct_preds/total_preds if total_preds > 0 else 0
    return accuracy,avg_loss

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Hauptteil ---
if __name__ == "__main__":
    set_seed(SEED)

    input_size = 12  # x_t(1) + instruction_oh(8) + R0_tm1(1) + R1_tm1(1) + y_tm1(1)
    output_size = 1 # y_t
    learning_rate = 0.001 # Potenziell niedrigere LR für komplexere Aufgabe
    batch_size = 512
    epochs = 200     # Mehr Epochen, da sehr komplex
    noise_level_data = 0.02 # Sehr niedriges Rauschen, um die Logik nicht zu stark zu stören
    target_accuracy_threshold = 0.90 # Ziel gesenkt, da die Aufgabe sehr schwer ist

    # Modellkonfigurationen
    dpp_units = 32 # Mehr Units für die komplexe Logik und größeren Input
    shared_g_dim_config = 6 # input_size / 2 = 6

    print(f"Task: Einfacher Instruktions-Interpreter")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 10000 # Viel mehr Daten
    num_test_sequences_per_epoch = 2000
    seq_len_task = 30 # Lange Programme/Sequenzen

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = InstructionInterpreterDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data)
    test_dataset = InstructionInterpreterDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss() # Bleibt für binären Output y_t

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte ~1167 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.2, min_lr=1e-7, verbose=True)


    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Instruktions-Interpreter ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_Interpreter", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Instruktions-Interpreter) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_Interpreter", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_Interpreter - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion (sehr grundlegend, da die Logik komplex ist)
    print("\n--- Alpha-Inspektion für einige Test-Samples (Instruktions-Interpreter) ---")
    model_to_train.eval()
    inspector_dataset = InstructionInterpreterDataset(1, max_inspect + 5, 0.0) # Saubere Daten für Inspektion
    inspector_loader = DataLoader(inspector_dataset, batch_size=1, shuffle=False)
    sample_count_inspect = 0
    max_inspect = 20

    instr_names = ["LOAD_R0_X", "LOAD_R1_X", "XOR_R0_R1", "AND_R0_R1", "NOT_R0", "OUT_R0", "OUT_R1", "NO_OP"]
    print(f"Format Input: [x_t, instr_oh(8), R0(t-1), R1(t-1), y(t-1)]")
    print("----------------------------------------------------------------------------------")

    with torch.no_grad():
        for insp_input_vec, insp_target_vec in inspector_loader:
            if sample_count_inspect >= max_inspect: break

            inp_tensor = insp_input_vec.to(DEVICE)
            _ = model_to_train(inp_tensor, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue

            inp_numpy = insp_input_vec[0].cpu().numpy() # Batch size ist 1
            lbl_numpy = insp_target_vec[0].cpu().item()
            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            x_disp = int(round(inp_numpy[0]))
            instr_oh_part = inp_numpy[1:9]
            instr_idx_disp = np.argmax(instr_oh_part)
            instr_name_disp = instr_names[instr_idx_disp]
            r0_tm1_disp = int(round(inp_numpy[9]))
            r1_tm1_disp = int(round(inp_numpy[10]))
            y_tm1_disp = int(round(inp_numpy[11]))

            print(f"S{sample_count_inspect+1}: In(x,Instr,R0,R1,y-1)~[{x_disp},{instr_name_disp}({instr_idx_disp}),{r0_tm1_disp},{r1_tm1_disp},{y_tm1_disp}], Target:{lbl_numpy:.0f}, MeanAlpha:{mean_alpha:.3f}")
            sample_count_inspect += 1
    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")