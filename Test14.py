import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 54 # Neuer Seed für diesen "extremen" Test
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
        # Hinzufügen einer weiteren normalen Schicht nach dem DPP-Layer könnte helfen
        # self.fc_intermediate = nn.Linear(hidden_size_dpp, hidden_size_dpp // 2)
        # self.relu_intermediate = nn.ReLU()
        # self.fc_out = nn.Linear(hidden_size_dpp // 2, output_size)
        self.relu1 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size_dpp, output_size)
        self.last_alphas = None

    def forward(self, x, return_alpha_flag=False):
        if return_alpha_flag and hasattr(self.dpp_layer1, 'forward') and \
           'return_alpha' in self.dpp_layer1.forward.__code__.co_varnames:
            out, alphas = self.dpp_layer1(x, return_alpha=True)
            self.last_alphas = alphas
        else:
            out = self.dpp_layer1(x, return_alpha=False)
        out = self.relu1(out)
        # if hasattr(self, 'fc_intermediate'):
        #    out = self.fc_intermediate(out)
        #    out = self.relu_intermediate(out)
        out = self.fc_out(out)
        return out

# --- Datengenerierung für "Mini-CPU V2" ---
class ExtremeCPUInterpreterDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, num_instructions=13, mem_size=2):
        super(ExtremeCPUInterpreterDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.mem_size = mem_size
        assert self.mem_size >= 2, "Memory size must be at least 2 for current instructions"

        # Instruktions-Indizes
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.LOAD_R2_X = 2  # R2 ist Adressregister (0 oder 1 für Mem[0] oder Mem[1])
        self.MOVE_R0_R1 = 3 # R0 <- R1
        self.XOR_R0_R1_R0 = 4 # R0 <- R0 XOR R1
        self.ADD_R0_R1_R0 = 5 # R0 <- (R0 + R1) % 2
        self.STORE_R0_MEM_AT_R2 = 6 # Mem[R2] <- R0
        self.LOAD_R0_FROM_MEM_AT_R2 = 7 # R0 <- Mem[R2]
        self.JUMP_IF_R0_ZERO_SKIP_2 = 8 # Wenn R0=0, PC += 2 (nächsten 2 Instr. überspringen)
        self.SET_FLAG_IF_R0_GT_R1 = 9  # F0 <- (R0 > R1)  (R0=1,R1=0)
        self.NOT_R0_IF_F0 = 10 # IF F0: R0 <- NOT R0
        self.OUT_R0 = 11
        self.NO_OP = 12

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r0, r1, r2, f0 = 0.0, 0.0, 0.0, 0.0
            memory = [0.0] * self.mem_size
            previous_y = 0.0
            
            program_counter = 0
            skip_counter = 0 # Für JUMP

            while program_counter < self.seq_len:
                if skip_counter > 0:
                    # Simuliere NO_OP für übersprungene Schritte
                    instr_idx = self.NO_OP
                    current_x1_bit = 0.0 # Irrelevant
                    skip_counter -=1
                else:
                    current_x1_bit = float(np.random.randint(0, 2))
                    instr_idx = np.random.randint(0, self.num_instructions)
                
                instruction_one_hot = self._to_one_hot(instr_idx, self.num_instructions)

                x1_t_noisy = current_x1_bit + np.random.normal(0, self.noise_level)
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                r2_tm1_noisy = r2 + np.random.normal(0, self.noise_level)
                f0_tm1_noisy = f0 + np.random.normal(0, self.noise_level)
                mem_tm1_noisy = [m + np.random.normal(0, self.noise_level) for m in memory]
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                input_features = [x1_t_noisy] + list(instruction_one_hot) + \
                                 [r0_tm1_noisy, r1_tm1_noisy, r2_tm1_noisy, f0_tm1_noisy] + \
                                 mem_tm1_noisy + [y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                next_r0, next_r1, next_r2, next_f0 = r0, r1, r2, f0
                next_memory = list(memory)
                target_y = 0.0 

                # Führe Instruktion aus (auf sauberen Werten für die Zustandsänderung)
                mem_addr = int(r2) % self.mem_size # Adresse für Speicherzugriff

                if instr_idx == self.LOAD_R0_X: next_r0 = current_x1_bit
                elif instr_idx == self.LOAD_R1_X: next_r1 = current_x1_bit
                elif instr_idx == self.LOAD_R2_X: next_r2 = current_x1_bit # Lade Adresse (0 oder 1)
                elif instr_idx == self.MOVE_R0_R1: next_r0 = r1
                elif instr_idx == self.XOR_R0_R1_R0: next_r0 = float(int(r0) ^ int(r1))
                elif instr_idx == self.ADD_R0_R1_R0: next_r0 = float(int(r0) ^ int(r1)) # Gleich wie XOR für Bits
                elif instr_idx == self.STORE_R0_MEM_AT_R2: next_memory[mem_addr] = r0
                elif instr_idx == self.LOAD_R0_FROM_MEM_AT_R2: next_r0 = memory[mem_addr]
                elif instr_idx == self.JUMP_IF_R0_ZERO_SKIP_2:
                    if int(r0) == 0: skip_counter = 2 
                elif instr_idx == self.SET_FLAG_IF_R0_GT_R1:
                    next_f0 = 1.0 if int(r0) == 1 and int(r1) == 0 else 0.0
                elif instr_idx == self.NOT_R0_IF_F0:
                    if int(f0) == 1: next_r0 = float(1 - int(r0))
                elif instr_idx == self.OUT_R0: target_y = next_r0 
                elif instr_idx == self.NO_OP: target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r0, r1, r2, f0 = next_r0, next_r1, next_r2, next_f0
                memory = next_memory
                previous_y = target_y
                program_counter += 1
                
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

    # Korrigierte Input-Größe
    # x1_t(1) + instr_oh(13) + R0(1) + R1(1) + R2(1) + F0(1) + Mem0(1) + Mem1(1) + y_tm1(1)
    input_size = 1 + 13 + 3 + 1 + 2 + 1 # = 21. Mit mem_size=2
    # Wenn mem_size=4 wäre, dann 1+13+3+1+4+1 = 23
    
    num_instructions_task = 13 # Anzahl der Instruktionen im Set
    mem_size_task = 2
    input_size = 1 + num_instructions_task + 3 + 1 + mem_size_task + 1 # x1, instr, R0-2, F0, Mem, y_t-1

    output_size = 1
    learning_rate = 0.0015 # Etwas niedriger starten
    batch_size = 1024
    epochs = 10      # erstmal wenig
    noise_level_data = 0.005 # Sehr niedriges Rauschen
    target_accuracy_threshold = 0.70 # Noch konservativer, da sehr komplex

    # Modellkonfigurationen
    dpp_units = 64 
    shared_g_dim_config = 11 # input_size (~22) / 2

    print(f"Task: Mini-CPU V2 (Registers, Speicher, Flag, Conditional Jump)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 40000 # Viele Daten
    num_test_sequences_per_epoch = 8000
    seq_len_task = 50 # Lange Programme

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = ExtremeCPUInterpreterDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, mem_size_task)
    test_dataset = ExtremeCPUInterpreterDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, mem_size_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte um die 4030 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-6)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.25, min_lr=1e-7, verbose=True) 

    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Mini-CPU V2 ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_MiniCPU_V2", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Mini-CPU V2) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_MiniCPU_V2", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_MiniCPU_V2 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Mini-CPU V2) ---")
    model_to_train.eval()
    max_inspect_alpha_cpu_v2 = 25 # Mehr Samples für Inspektion

    # Saubere Daten für Inspektion
    inspector_dataset_cpu_v2 = ExtremeCPUInterpreterDataset(1, max_inspect_alpha_cpu_v2 + seq_len_task, 0.0, num_instructions_task, mem_size_task)
    inspector_loader_cpu_v2 = DataLoader(inspector_dataset_cpu_v2, batch_size=1, shuffle=False)
    sample_count_inspect = 0

    instr_names_cpu_v2 = [ # Angepasst an 13 Instruktionen
        "LOAD_R0_X", "LOAD_R1_X", "LOAD_R2_X", "MOVE_R0_R1", 
        "XOR_R0_R1_R0", "ADD_R0_R1_R0", "STO_R0_M[R2]", "LOAD_R0_M[R2]",
        "JMP_IF_R0_0", "SET_F0_R0>R1", "NOT_R0_IF_F0", "OUT_R0", "NO_OP"
    ]
    print(f"Format Input: [x1, instr_oh({num_instructions_task}), R0,R1,R2, F0, M0,M1, y-1]") # Angepasst
    print("----------------------------------------------------------------------------------")

    with torch.no_grad():
        # Zustandsvariablen für die Inspektion, um den "echten" Zustand nachzuvollziehen
        insp_r0, insp_r1, insp_r2, insp_f0 = 0.0, 0.0, 0.0, 0.0
        insp_memory = [0.0] * mem_size_task
        insp_previous_y = 0.0
        insp_skip_counter = 0

        for insp_idx, (insp_input_vec_model, insp_target_vec_model) in enumerate(inspector_loader_cpu_v2):
            if sample_count_inspect >= max_inspect_alpha_cpu_v2: break

            # Die insp_input_vec_model sind die (verrauschten) Inputs, die das Modell im Dataset sehen würde.
            # Für die Anzeige wollen wir die "sauberen" Werte, die zu diesem Schritt geführt haben.
            # Der Datengenerator muss die sauberen Werte intern verwalten.

            # Erzeuge die Inputs, die das Modell sieht, basierend auf unseren insp_ Zuständen
            current_insp_x1 = float(random.randint(0,1)) # Zufälliges x1 für diesen Inspektionsschritt
            
            if insp_skip_counter > 0:
                current_instr_idx = 12 # NO_OP
                insp_skip_counter -= 1
            else:
                current_instr_idx = random.randint(0, num_instructions_task -1)

            current_instr_oh = np.zeros(num_instructions_task, dtype=np.float32)
            current_instr_oh[current_instr_idx] = 1.0

            # Das sind die Inputs, die das Modell *sehen* würde (sauber für Inspektion)
            model_input_features = [current_insp_x1] + list(current_instr_oh) + \
                                   [insp_r0, insp_r1, insp_r2, insp_f0] + \
                                   list(insp_memory) + [insp_previous_y]
            inp_tensor = torch.tensor([model_input_features], dtype=torch.float32).to(DEVICE)


            _ = model_to_train(inp_tensor, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue
            
            model_output_logit = model_to_train(inp_tensor) # Erneuter Forward Pass ohne Alpha-Return für Output
            model_output_y = float(torch.sigmoid(model_output_logit[0]).item() > 0.5)


            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            # Berechne den korrekten nächsten Zustand und Ziel-y basierend auf sauberen Werten
            # (Wiederholung der Logik aus dem Dataset-Generator für den aktuellen Schritt)
            next_insp_r0, next_insp_r1, next_insp_r2, next_insp_f0 = insp_r0, insp_r1, insp_r2, insp_f0
            next_insp_memory = list(insp_memory)
            correct_target_y = 0.0
            mem_addr_insp = int(insp_r2) % mem_size_task

            if current_instr_idx == 0: next_insp_r0 = current_insp_x1
            elif current_instr_idx == 1: next_insp_r1 = current_insp_x1
            elif current_instr_idx == 2: next_insp_r2 = current_insp_x1
            elif current_instr_idx == 3: next_insp_r0 = insp_r1
            elif current_instr_idx == 4: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1))
            elif current_instr_idx == 5: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1)) # ADD ist XOR für Bits
            elif current_instr_idx == 6: next_insp_memory[mem_addr_insp] = insp_r0
            elif current_instr_idx == 7: next_insp_r0 = insp_memory[mem_addr_insp]
            elif current_instr_idx == 8:
                if int(insp_r0) == 0: insp_skip_counter = 2 # Setze Skip für die *nächsten* Iterationen des äußeren Loops
            elif current_instr_idx == 9:
                next_insp_f0 = 1.0 if int(insp_r0) == 1 and int(insp_r1) == 0 else 0.0
            elif current_instr_idx == 10:
                if int(insp_f0) == 1: next_insp_r0 = float(1 - int(insp_r0))
            elif current_instr_idx == 11: correct_target_y = next_insp_r0 # Output ist der *neue* R0 Wert
            elif current_instr_idx == 12: correct_target_y = insp_previous_y
            
            # Für Instruktionen, die keinen direkten Output y_t haben, ist correct_target_y 0 (oder was im Dataset definiert wurde)
            # Für OUT_R0 haben wir correct_target_y = next_insp_r0 gesetzt.

            print(f"S{sample_count_inspect+1}: x1:{int(current_insp_x1)}, Instr:{instr_names_cpu_v2[current_instr_idx]}({current_instr_idx}) | R0i:{int(insp_r0)},R1i:{int(insp_r1)},R2i:{int(insp_r2)},F0i:{int(insp_f0)},M0i:{int(insp_memory[0])},M1i:{int(insp_memory[1])},y-1i:{int(insp_previous_y)} | Pred_y:{int(model_output_y)}, Target_y:{int(correct_target_y)}, MeanAlpha:{mean_alpha:.3f}")
            sample_count_inspect += 1

            # Update Zustände für den nächsten Inspektionsschritt
            insp_r0, insp_r1, insp_r2, insp_f0 = next_insp_r0, next_insp_r1, next_insp_r2, next_insp_f0
            insp_memory = next_insp_memory
            insp_previous_y = correct_target_y # Wichtig: Verwende den korrekten Zielwert als nächsten y-1

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")