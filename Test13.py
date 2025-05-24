import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 53 # Neuer Seed für diesen sehr komplexen Test
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

# --- Datengenerierung für "Mini-CPU V1" ---
class MiniCPUInterpreterDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, num_instructions=10):
        super(MiniCPUInterpreterDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions

        # Instruktions-Indizes
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.MOVE_R0_R1 = 2 # R0 <- R1
        self.XOR_R0_R1_R0 = 3 # R0 <- R0 XOR R1
        self.AND_R0_R1_R0 = 4 # R0 <- R0 AND R1
        self.SET_FLAG_IF_R0_EQ_R1 = 5 # F0 <- (R0 == R1)
        self.NOT_R0_IF_F0 = 6 # IF F0: R0 <- NOT R0
        self.OUT_R0 = 7
        self.OUT_R1 = 8
        self.NO_OP = 9

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r0 = 0.0
            r1 = 0.0
            f0 = 0.0 # Flag-Register
            previous_y = 0.0

            for t in range(self.seq_len):
                current_x1_bit = float(np.random.randint(0, 2)) # Datenbit für LOAD Instruktionen
                instr_idx = np.random.randint(0, self.num_instructions)
                instruction_one_hot = self._to_one_hot(instr_idx, self.num_instructions)

                # Inputs für das Modell (mit Rauschen)
                x1_t_noisy = current_x1_bit + np.random.normal(0, self.noise_level)
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                f0_tm1_noisy = f0 + np.random.normal(0, self.noise_level)
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                input_features = [x1_t_noisy] + list(instruction_one_hot) + \
                                 [r0_tm1_noisy, r1_tm1_noisy, f0_tm1_noisy, y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Temporäre Register/Flag für diesen Schritt (basierend auf sauberen Werten von t-1)
                next_r0 = r0
                next_r1 = r1
                next_f0 = f0 # Flag wird auch für den nächsten Schritt vorbereitet
                target_y = 0.0 # Default für nicht-Output Instruktionen

                # Instruktionsausführung (Logik für Zustandsänderung und Target-Generierung)
                if instr_idx == self.LOAD_R0_X:
                    next_r0 = current_x1_bit
                elif instr_idx == self.LOAD_R1_X:
                    next_r1 = current_x1_bit
                elif instr_idx == self.MOVE_R0_R1:
                    next_r0 = r1 # Wert von R1 aus t-1
                elif instr_idx == self.XOR_R0_R1_R0:
                    next_r0 = float(int(r0) ^ int(r1))
                elif instr_idx == self.AND_R0_R1_R0:
                    next_r0 = float(int(r0) & int(r1))
                elif instr_idx == self.SET_FLAG_IF_R0_EQ_R1:
                    next_f0 = 1.0 if int(r0) == int(r1) else 0.0
                elif instr_idx == self.NOT_R0_IF_F0:
                    if int(f0) == 1: # Prüfe Flag von t-1
                        next_r0 = float(1 - int(r0))
                    # else: R0 bleibt unverändert (next_r0 = r0)
                elif instr_idx == self.OUT_R0:
                    target_y = next_r0 # Wert von R0 *nach* potenzieller Änderung in diesem Schritt
                elif instr_idx == self.OUT_R1:
                    target_y = next_r1 # Wert von R1 *nach* potenzieller Änderung in diesem Schritt
                elif instr_idx == self.NO_OP:
                    target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                # Update der "echten" Zustände für den nächsten Input-Zeitschritt
                r0 = next_r0
                r1 = next_r1
                f0 = next_f0
                previous_y = target_y

    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen mit AMP ---
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
                scheduler_obj.step(accuracy) 
            else:
                scheduler_obj.step()

        if (accuracy>=target_accuracy and epoch_at_target_accuracy is None) or (epoch+1)%(epochs//10 if epochs>=10 else 1)==0 or epoch==0 or epoch==epochs-1 :
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

    input_size = 15  # x1_t(1) + instr_oh(10) + R0_tm1(1) + R1_tm1(1) + F0_tm1(1) + y_tm1(1)
    output_size = 1 # y_t
    learning_rate = 0.002
    batch_size = 1024 
    epochs = 10 # nur 10 Epochen
    noise_level_data = 0.01
    target_accuracy_threshold = 0.80 # Sehr ambitioniertes Ziel für diese Komplexität
    num_instructions_task = 10

    # Modellkonfigurationen
    dpp_units = 48 
    shared_g_dim_config = 8 # (15 Input Features / 2) ~ 7-8

    print(f"Task: Mini-CPU V1 (Registers, Flag, Conditional Op)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 30000 # Noch mehr Daten
    num_test_sequences_per_epoch = 6000
    seq_len_task = 40 # Längere Programme

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = MiniCPUInterpreterDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task)
    test_dataset = MiniCPUInterpreterDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte 2145 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-6)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25, factor=0.3, min_lr=1e-7, verbose=True) 

    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Mini-CPU V1 ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_MiniCPU_V1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Mini-CPU V1) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_MiniCPU_V1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_MiniCPU_V1 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Mini-CPU V1) ---")
    model_to_train.eval()
    max_inspect_alpha_cpu = 20 # Variable für Alpha-Inspektion definiert

    inspector_dataset_cpu = MiniCPUInterpreterDataset(1, max_inspect_alpha_cpu + seq_len_task, 0.0, num_instructions_task) # Saubere Daten
    inspector_loader_cpu = DataLoader(inspector_dataset_cpu, batch_size=1, shuffle=False)
    sample_count_inspect = 0

    instr_names_cpu = [
        "LOAD_R0_X", "LOAD_R1_X", "MOVE_R0_R1", "XOR_R0_R1_R0", "AND_R0_R1_R0",
        "SET_FLAG_IF_R0_EQ_R1", "NOT_R0_IF_F0", "OUT_R0", "OUT_R1", "NO_OP"
    ]
    print(f"Format Input: [x1, instr_oh({num_instructions_task}), R0_tm1, R1_tm1, F0_tm1, y_tm1]")
    print("----------------------------------------------------------------------------------")

    with torch.no_grad():
        for insp_idx, (insp_input_vec, insp_target_vec) in enumerate(inspector_loader_cpu):
            if sample_count_inspect >= max_inspect_alpha_cpu: break

            inp_tensor = insp_input_vec.to(DEVICE)
            _ = model_to_train(inp_tensor, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue

            inp_numpy = insp_input_vec[0].cpu().numpy()
            lbl_numpy = insp_target_vec[0].cpu().item()
            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            x1_disp = int(round(inp_numpy[0]))
            instr_oh_part = inp_numpy[1 : 1 + num_instructions_task]
            instr_idx_disp = np.argmax(instr_oh_part)
            instr_name_disp = instr_names_cpu[instr_idx_disp]
            
            r0_tm1_disp = int(round(inp_numpy[1+num_instructions_task]))
            r1_tm1_disp = int(round(inp_numpy[1+num_instructions_task+1]))
            f0_tm1_disp = int(round(inp_numpy[1+num_instructions_task+2]))
            y_tm1_disp = int(round(inp_numpy[1+num_instructions_task+3]))

            print(f"S{sample_count_inspect+1}: x1:{x1_disp}, Instr:{instr_name_disp}({instr_idx_disp}) | R0i:{r0_tm1_disp},R1i:{r1_tm1_disp},F0i:{f0_tm1_disp},y-1i:{y_tm1_disp} | Target_y:{lbl_numpy:.0f}, MeanAlpha:{mean_alpha:.3f}")
            sample_count_inspect += 1

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")