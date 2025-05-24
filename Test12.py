import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 52 # Neuer Seed für diesen sehr komplexen Test
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

# --- Datengenerierung für "Erweiterter Instruktions-Interpreter mit Speicher" ---
class AdvInstructionInterpreterDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, num_instructions=12, mem_size=2):
        super(AdvInstructionInterpreterDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.mem_size = mem_size # Größe des Speichers

        # Instruktions-Indizes (für Klarheit)
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.XOR_R0_R1_STORE_R0 = 2
        self.AND_R0_R1_STORE_R0 = 3
        self.STORE_R0_TO_MEM0 = 4
        self.STORE_R1_TO_MEM1 = 5
        self.LOAD_R0_FROM_MEM0 = 6
        self.LOAD_R1_FROM_MEM1 = 7
        self.ADD_MEM0_MEM1_STORE_R0 = 8 # (Mem[0] + Mem[1]) % 2 -> R0
        self.OUT_R0 = 9
        self.OUT_MEM0 = 10 # Geändert von OUT_R1 zu OUT_MEM0 für mehr Speichertest
        self.NO_OP = 11


    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            # Register, Speicher und vorheriger Output
            r0 = 0.0
            r1 = 0.0
            memory = [0.0] * self.mem_size
            previous_y = 0.0

            for t in range(self.seq_len):
                current_x_bit = float(np.random.randint(0, 2))
                instr_idx = np.random.randint(0, self.num_instructions)
                instruction_one_hot = self._to_one_hot(instr_idx, self.num_instructions)

                # Inputs für das Modell (mit Rauschen)
                x_t_noisy = current_x_bit + np.random.normal(0, self.noise_level)
                
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                mem_tm1_noisy = [m + np.random.normal(0, self.noise_level) for m in memory]
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                input_features = [x_t_noisy] + list(instruction_one_hot) + \
                                 [r0_tm1_noisy, r1_tm1_noisy] + mem_tm1_noisy + [y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Zielberechnung y_t und Register/Speicher-Updates (basierend auf sauberen Werten)
                current_r0 = r0 # Temporäre Kopien für diesen Schritt
                current_r1 = r1
                current_memory = list(memory) # Kopie
                target_y = 0.0 # Default für nicht-Output Instruktionen

                if instr_idx == self.LOAD_R0_X:
                    current_r0 = current_x_bit
                elif instr_idx == self.LOAD_R1_X:
                    current_r1 = current_x_bit
                elif instr_idx == self.XOR_R0_R1_STORE_R0:
                    current_r0 = float(int(r0) ^ int(r1))
                elif instr_idx == self.AND_R0_R1_STORE_R0:
                    current_r0 = float(int(r0) & int(r1))
                elif instr_idx == self.STORE_R0_TO_MEM0:
                    if self.mem_size > 0: current_memory[0] = r0
                elif instr_idx == self.STORE_R1_TO_MEM1:
                    if self.mem_size > 1: current_memory[1] = r1
                elif instr_idx == self.LOAD_R0_FROM_MEM0:
                    if self.mem_size > 0: current_r0 = memory[0]
                elif instr_idx == self.LOAD_R1_FROM_MEM1:
                    if self.mem_size > 1: current_r1 = memory[1]
                elif instr_idx == self.ADD_MEM0_MEM1_STORE_R0:
                    if self.mem_size > 1:
                        current_r0 = float(int(memory[0]) ^ int(memory[1])) # (m0+m1)%2 ist XOR
                elif instr_idx == self.OUT_R0:
                    target_y = current_r0 
                elif instr_idx == self.OUT_MEM0:
                    if self.mem_size > 0: target_y = current_memory[0]
                elif instr_idx == self.NO_OP:
                    target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                # Update der "echten" Zustände für den nächsten Input-Zeitschritt
                r0 = current_r0
                r1 = current_r1
                memory = current_memory
                previous_y = target_y

    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen mit AMP (unverändert, außer Scheduler im Loop) ---
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
            else: # Für CosineAnnealing etc. pro Epoche
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

    input_size = 18  # x_t(1) + instr_oh(12) + R0(1) + R1(1) + Mem0(1) + Mem1(1) + y_tm1(1)
    output_size = 1 # y_t
    learning_rate = 0.001 
    batch_size = 1024 # Größere Batch Size, da mehr Daten pro Epoche
    epochs = 15      # Mehr Epochen für diese sehr komplexe Aufgabe
    noise_level_data = 0.01 # Sehr, sehr niedriges Rauschen
    target_accuracy_threshold = 0.85 # Ziel gesenkt, da die Aufgabe extrem schwer ist
    num_instructions_task = 12
    mem_size_task = 2

    # Modellkonfigurationen
    dpp_units = 48 # Erhöhte Kapazität
    shared_g_dim_config = 9 # input_size / 2 = 9

    print(f"Task: Erweiterter Instruktions-Interpreter mit Speicher (MemSize={mem_size_task})")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 20000 # Deutlich mehr Daten
    num_test_sequences_per_epoch = 4000
    seq_len_task = 35 # Lange Programme/Sequenzen

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = AdvInstructionInterpreterDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, mem_size_task)
    test_dataset = AdvInstructionInterpreterDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, mem_size_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0) # num_workers=0 für Stabilität mit IterableDataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte ~2600 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-6) # Noch weniger WD
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    # Reduziere LR langsamer, gib dem Modell mehr Zeit mit höherer LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, factor=0.3, min_lr=5e-7, verbose=True) 


    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Erw. Instruktions-Interpreter ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_AdvInterpreter", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Erw. Instruktions-Interpreter) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_AdvInterpreter", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_AdvInterpreter - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Erw. Instruktions-Interpreter) ---")
    model_to_train.eval()
    max_inspect_alpha = 20 # Variable für Alpha-Inspektion definiert

    # Saubere Daten für Inspektion
    inspector_dataset = AdvInstructionInterpreterDataset(1, max_inspect_alpha + seq_len_task, 0.0, num_instructions_task, mem_size_task)
    inspector_loader = DataLoader(inspector_dataset, batch_size=1, shuffle=False)
    sample_count_inspect = 0

    instr_names_adv = [
        "LOAD_R0_X", "LOAD_R1_X", "XOR_R0_R1", "AND_R0_R1",
        "STO_R0_M0", "STO_R1_M1", "LOAD_R0_M0", "LOAD_R1_M1",
        "ADD_M0_M1_R0", "OUT_R0", "OUT_MEM0", "NO_OP"
    ]
    print(f"Format Input: [x, instr_oh({num_instructions_task}), R0, R1, M0, M1, y-1]")
    print("----------------------------------------------------------------------------------")

    # Lokale Variablen für die Zustandsverfolgung in der Inspektion
    insp_r0, insp_r1 = 0.0, 0.0
    insp_mem = [0.0] * mem_size_task
    insp_prev_y = 0.0

    with torch.no_grad():
        for insp_idx, (insp_input_vec, insp_target_vec) in enumerate(inspector_loader):
            if sample_count_inspect >= max_inspect_alpha: break

            inp_tensor = insp_input_vec.to(DEVICE) # Batch size ist 1
            _ = model_to_train(inp_tensor, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue

            inp_numpy = insp_input_vec[0].cpu().numpy()
            lbl_numpy = insp_target_vec[0].cpu().item()
            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            x_disp = int(round(inp_numpy[0]))
            instr_oh_part = inp_numpy[1 : 1 + num_instructions_task]
            instr_idx_disp = np.argmax(instr_oh_part)
            instr_name_disp = instr_names_adv[instr_idx_disp]
            
            # Diese sind die Inputs des Modells, also die Zustände *vor* der aktuellen Instruktion
            r0_tm1_disp_model_sees = int(round(inp_numpy[1+num_instructions_task]))
            r1_tm1_disp_model_sees = int(round(inp_numpy[1+num_instructions_task+1]))
            mem0_tm1_disp_model_sees = int(round(inp_numpy[1+num_instructions_task+2]))
            mem1_tm1_disp_model_sees = int(round(inp_numpy[1+num_instructions_task+3]))
            y_tm1_disp_model_sees = int(round(inp_numpy[1+num_instructions_task+4]))


            # Ausgabe der Zustände *bevor* die aktuelle Instruktion ausgeführt wurde (wie das Modell sie sieht)
            print(f"S{sample_count_inspect+1}: x:{x_disp}, Instr:{instr_name_disp}({instr_idx_disp}) | R0_in:{r0_tm1_disp_model_sees}, R1_in:{r1_tm1_disp_model_sees}, M0_in:{mem0_tm1_disp_model_sees}, M1_in:{mem1_tm1_disp_model_sees}, y-1_in:{y_tm1_disp_model_sees} | Target_y:{lbl_numpy:.0f}, MeanAlpha:{mean_alpha:.3f}")
            sample_count_inspect += 1
            
            # Die Zustandsaktualisierung der Datengenerierung für das nächste Sample (wird nicht direkt für den Print verwendet, aber um sicherzustellen, dass der Loader die Sequenz fortsetzt)
            # Dieser Teil ist für die reine Inspektion des aktuellen Schritts nicht nötig, aber der Loader muss weiterlaufen.

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")