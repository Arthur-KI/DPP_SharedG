import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 56 # Beibehalten vom letzten Versuch für diesen Test
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
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w); bound = 1/(fan_in**0.5) if fan_in > 0 else 0; nn.init.uniform_(b, -bound, bound)
        nn.init.kaiming_uniform_(self.w_g_shared, a=(5**0.5))
        if self.b_g_shared is not None:
            fan_in_g_shared, _ = nn.init._calculate_fan_in_and_fan_out(self.w_g_shared); bound_g_shared = 1/(fan_in_g_shared**0.5) if fan_in_g_shared > 0 else 0; nn.init.uniform_(self.b_g_shared, -bound_g_shared, bound_g_shared)
        nn.init.kaiming_uniform_(self.w_g_unit, a=(5**0.5))
        if self.b_g_unit is not None:
            fan_in_g_unit, _ = nn.init._calculate_fan_in_and_fan_out(self.w_g_unit); bound_g_unit = 1/(fan_in_g_unit**0.5) if fan_in_g_unit > 0 else 0; nn.init.uniform_(self.b_g_unit, -bound_g_unit, bound_g_unit)

    def forward(self, x, return_alpha=False):
        z_a = F.linear(x, self.w_a, self.b_a); z_b = F.linear(x, self.w_b, self.b_b)
        x_shared_g = F.linear(x, self.w_g_shared, self.b_g_shared)
        g_logits = F.linear(x_shared_g, self.w_g_unit, self.b_g_unit)
        alpha = torch.sigmoid(g_logits); z_final = alpha * z_a + (1 - alpha) * z_b
        if return_alpha: return z_final, alpha
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

# --- Datengenerierung für "Stack-Maschine V1" ---
class StackMachineV1Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 num_instructions=13, data_mem_size=4, ret_stack_depth=4): # num_instructions ist jetzt 13
        super(StackMachineV1Dataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.data_mem_size = data_mem_size
        self.ret_stack_depth = ret_stack_depth

        # Instruktions-Indizes (13 Instruktionen)
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.LOAD_RA_X = 2      # RA (Adressregister) mit x1_t laden (0 oder 1 für Mem[0] oder Mem[1] bei mem_size=2, oder 0..3 bei mem_size=4)
        self.MOVE_R0_R1 = 3     # R0 <- R1
        self.XOR_R0_R1_R0 = 4   # R0 <- R0 XOR R1
        # self.ADD_R0_R1_R0 = 5 # Entfernt, da XOR für Bits dasselbe ist und wir Platz sparen
        self.NOT_R0 = 5         # Neuer Index
        self.STORE_R0_MEM_AT_RA = 6
        self.LOAD_R0_FROM_MEM_AT_RA = 7
        self.PUSH_PC_CALL_SUB = 8   # Eine generische CALL-Instruktion, Ziel wird unten definiert
        self.RETURN = 9
        self.SET_FLAG_IF_R0_ZERO = 10
        self.JUMP_IF_F0_SKIP_N = 11 # N ist fest (z.B. 2)
        self.OUT_R0 = 12
        # self.OUT_STACK_TOP entfernt für Einfachheit
        self.NO_OP = self.num_instructions -1 # Sicherstellen, dass NO_OP der letzte Index ist, wenn num_instr 13 ist, ist das 12

        if self.NO_OP != 12: # Sicherheitscheck, falls num_instructions geändert wird
             print(f"WARNUNG: NO_OP Index ({self.NO_OP}) stimmt nicht mit Erwartung (12) überein. Überprüfe Instruktionsdefinitionen!")


    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes:
            one_hot[int(value)] = 1.0
        else: # Fallback für ungültige Werte (z.B. SP bei leerem Stack)
            # print(f"Warnung: Ungültiger Wert {value} für One-Hot mit {num_classes} Klassen. Setze auf Null-Vektor.")
            pass # Bleibt Null-Vektor
        return one_hot

    def __iter__(self):
        SUBROUTINE_1_ADDR = self.seq_len + 5 # Feste Adresse für eine simulierte Subroutine

        for _ in range(self.num_sequences_per_epoch):
            r0, r1, ra, f0 = 0.0, 0.0, 0.0, 0.0
            data_memory = [0.0] * self.data_mem_size
            ret_stack = [0] * self.ret_stack_depth
            ret_sp = 0
            previous_y = 0.0
            program_counter = 0
            skip_next_n_instr = 0

            idx_in_sequence = 0
            while idx_in_sequence < self.seq_len:
                current_x1_bit_clean = float(np.random.randint(0, 2))
                current_instr_idx_clean = 0

                if skip_next_n_instr > 0:
                    current_instr_idx_clean = self.NO_OP
                    skip_next_n_instr -= 1
                elif program_counter == SUBROUTINE_1_ADDR:
                    if idx_in_sequence >= self.seq_len: break
                    current_instr_idx_clean = self.NOT_R0 # Beispiel-Instruktion für Subroutine
                elif program_counter == SUBROUTINE_1_ADDR + 1:
                    if idx_in_sequence >= self.seq_len: break
                    current_instr_idx_clean = self.RETURN
                else:
                    current_instr_idx_clean = np.random.randint(0, self.num_instructions)
                    if current_instr_idx_clean == self.PUSH_PC_CALL_SUB and ret_sp >= self.ret_stack_depth:
                        current_instr_idx_clean = self.NO_OP
                    if current_instr_idx_clean == self.RETURN and ret_sp == 0:
                        current_instr_idx_clean = self.NO_OP
                
                instruction_one_hot = self._to_one_hot(current_instr_idx_clean, self.num_instructions)
                # RA als Wert (0..data_mem_size-1), nicht One-Hot, für Input-Vektor, um Features zu sparen
                # SP (Return Stack Pointer) als One-Hot
                sp_one_hot = self._to_one_hot(ret_sp % self.ret_stack_depth, self.ret_stack_depth)


                x1_t_noisy = current_x1_bit_clean + np.random.normal(0, self.noise_level)
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                ra_val_noisy = ra + np.random.normal(0, self.noise_level) # RA als Wert
                f0_tm1_noisy = f0 + np.random.normal(0, self.noise_level)
                data_mem_noisy = [m_val + np.random.normal(0, self.noise_level) for m_val in data_memory]
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                # Input: x1(1) + instr(13) + R0(1) + R1(1) + RA_val(1) + F0(1) + DataMem(4) + SP_oh(4) + y_tm1(1) = 26
                input_features = ([x1_t_noisy] + list(instruction_one_hot) +
                                  [r0_tm1_noisy, r1_tm1_noisy, ra_val_noisy, f0_tm1_noisy] +
                                  data_mem_noisy + list(sp_one_hot) + [y_tm1_noisy])
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                next_r0, next_r1, next_ra, next_f0 = r0, r1, ra, f0
                next_data_memory = list(data_memory)
                target_y = 0.0
                next_pc_val = program_counter + 1
                mem_addr_clean = int(ra) % self.data_mem_size

                if current_instr_idx_clean == self.LOAD_R0_X: next_r0 = current_x1_bit_clean
                elif current_instr_idx_clean == self.LOAD_R1_X: next_r1 = current_x1_bit_clean
                elif current_instr_idx_clean == self.LOAD_RA_X: next_ra = float(int(current_x1_bit_clean) % self.data_mem_size)
                elif current_instr_idx_clean == self.MOVE_R0_R1: next_r0 = r1
                elif current_instr_idx_clean == self.XOR_R0_R1_R0: next_r0 = float(int(r0) ^ int(r1))
                # ADD_R0_R1_R0 wurde entfernt, um auf 13 Instruktionen zu kommen
                elif current_instr_idx_clean == self.NOT_R0: next_r0 = float(1 - int(r0))
                elif current_instr_idx_clean == self.STORE_R0_MEM_AT_RA: next_data_memory[mem_addr_clean] = r0
                elif current_instr_idx_clean == self.LOAD_R0_FROM_MEM_AT_RA: next_r0 = data_memory[mem_addr_clean]
                elif current_instr_idx_clean == self.PUSH_PC_CALL_SUB:
                    if ret_sp < self.ret_stack_depth:
                        ret_stack[ret_sp] = program_counter + 1
                        ret_sp = (ret_sp + 1) # SP zeigt auf das nächste freie Element
                    next_pc_val = SUBROUTINE_1_ADDR # Springe immer zu SUB1 für diesen Test
                elif current_instr_idx_clean == self.RETURN:
                    if ret_sp > 0:
                        ret_sp = ret_sp - 1 # SP zeigt jetzt auf das oberste Element
                        next_pc_val = ret_stack[ret_sp]
                elif current_instr_idx_clean == self.SET_FLAG_IF_R0_ZERO:
                    next_f0 = 1.0 if int(r0) == 0 else 0.0
                elif current_instr_idx_clean == self.JUMP_IF_F0_SKIP_N:
                    if int(f0) == 1: skip_next_n_instr = 2
                elif current_instr_idx_clean == self.OUT_R0: target_y = next_r0
                elif current_instr_idx_clean == self.NO_OP: target_y = previous_y
                
                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r0, r1, ra, f0 = next_r0, next_r1, next_ra, next_f0
                data_memory = next_data_memory
                previous_y = target_y # WICHTIG: previous_y für nächsten Schritt ist der *berechnete* target_y
                program_counter = next_pc_val
                idx_in_sequence +=1
                
    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen (weitgehend unverändert) ---
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

        if (accuracy>=target_accuracy and epoch_at_target_accuracy is None) or (epoch+1)%(epochs//20 if epochs>=20 else 1)==0 or epoch==0 or epoch==epochs-1 :
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
    
    num_instructions_task = 13
    data_mem_size_task = 4
    ret_stack_depth_task = 4
    
    # Input: x1(1) + instr_oh(13) + R0(1) + R1(1) + RA_val(1) + F0(1) + DataMem(4) + SP_oh(4) + y_tm1(1) = 27
    input_size = 1 + num_instructions_task + 1 + 1 + 1 + 1 + data_mem_size_task + ret_stack_depth_task + 1

    output_size = 1
    learning_rate = 0.001 
    batch_size = 2048 
    epochs = 5      # Beibehalten oder anpassen
    noise_level_data = 0.001 
    target_accuracy_threshold = 0.65 # Beibehalten oder anpassen

    dpp_units = 64 # Beibehalten oder anpassen (z.B. 96)
    shared_g_dim_config = 13 # (input_size / 2) -> 27/2 = 13

    print(f"Task: Stack-Maschine V1 (Registers, Stack, Mem, Flag, Jumps, Calls)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    num_train_sequences_per_epoch = 60000 
    num_test_sequences_per_epoch = 12000
    seq_len_task = 60

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = StackMachineV1Dataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task)
    test_dataset = StackMachineV1Dataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}")

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-7)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=40, factor=0.25, min_lr=5e-8, verbose=True) 

    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Stack-Maschine V1 ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_StackMachineV1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Stack-Maschine V1) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_StackMachineV1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_StackMachineV1 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Stack-Maschine V1) ---")
    model_to_train.eval()
    max_inspect_alpha_sm_v1 = 30 

    inspector_dataset_sm_v1 = StackMachineV1Dataset(1, max_inspect_alpha_sm_v1 + seq_len_task, 0.0, num_instructions_task, data_mem_size_task, ret_stack_depth_task) # Saubere Daten
    inspector_loader_sm_v1 = DataLoader(inspector_dataset_sm_v1, batch_size=1, shuffle=False)
    sample_count_inspect = 0

    instr_names_sm_v1 = [ # 13 Instruktionen
        "LOAD_R0_X", "LOAD_R1_X", "LOAD_RA_X", "MOVE_R0_R1", "XOR_R0_R1_R0",
        "NOT_R0", "STO_R0_M[RA]", "LOAD_R0_M[RA]", "CALL_SUB", # ADD entfernt, PUSH_PC_CALL_SUB1/2 zu CALL_SUB
        "RETURN", "SET_F0_R0_0", "JMP_IF_F0_SK2", "OUT_R0", "NO_OP" # NO_OP ist Index 12
    ]
    if len(instr_names_sm_v1) != num_instructions_task: # Sicherstellen, dass die Namenliste passt
        print(f"WARNUNG: Länge der instr_names_sm_v1 ({len(instr_names_sm_v1)}) passt nicht zu num_instructions_task ({num_instructions_task})")
        instr_names_sm_v1 = [f"INSTR_{i}" for i in range(num_instructions_task)]
        if current_instr_idx_clean == 12 : # Sicherstellen, dass NO_OP korrekt adressiert wird, falls die Liste anders ist
           current_instr_idx_clean = num_instructions_task -1


    print(f"Format Input: [x1, instr_oh({num_instructions_task}), R0,R1,RA_val,F0, DataMem({data_mem_size_task}), SP_oh({ret_stack_depth_task}), y-1]")
    print("----------------------------------------------------------------------------------------------------")
    
    with torch.no_grad():
        insp_r0, insp_r1, insp_ra, insp_f0 = 0.0, 0.0, 0.0, 0.0
        insp_data_memory = [0.0] * data_mem_size_task
        insp_ret_stack = [0] * ret_stack_depth_task
        insp_ret_sp = 0
        insp_previous_y = 0.0
        insp_program_counter = 0
        insp_skip_next_n_instr = 0
        SUBROUTINE_1_ADDR_INSP_V2 = seq_len_task + 5 # Muss mit Dataset-Definition übereinstimmen


        for insp_idx in range(max_inspect_alpha_sm_v1): # Direkte Schleife über die Anzahl der Inspektionsschritte
            if sample_count_inspect >= max_inspect_alpha_sm_v1: break

            current_insp_x1_clean = float(random.randint(0,1))
            current_instr_idx_clean_insp = 0
            
            if insp_skip_next_n_instr > 0:
                current_instr_idx_clean_insp = num_instructions_task - 1 # NO_OP
                insp_skip_next_n_instr -= 1
            elif insp_program_counter == SUBROUTINE_1_ADDR_INSP_V2 : 
                current_instr_idx_clean_insp = 5 # NOT_R0 als Beispiel für Subroutine
            elif insp_program_counter == SUBROUTINE_1_ADDR_INSP_V2 + 1: 
                current_instr_idx_clean_insp = 9 # RETURN
            else:
                current_instr_idx_clean_insp = random.randint(0, num_instructions_task -1)
                if current_instr_idx_clean_insp == 8 and insp_ret_sp >= ret_stack_depth_task: # PUSH_PC_CALL_SUB
                    current_instr_idx_clean_insp = num_instructions_task - 1 # NO_OP
                if current_instr_idx_clean_insp == 9 and insp_ret_sp == 0: # RETURN
                    current_instr_idx_clean_insp = num_instructions_task - 1 # NO_OP

            current_instr_oh_clean = np.zeros(num_instructions_task, dtype=np.float32); current_instr_oh_clean[current_instr_idx_clean_insp] = 1.0
            current_sp_oh_clean = np.zeros(ret_stack_depth_task, dtype=np.float32); current_sp_oh_clean[insp_ret_sp % ret_stack_depth_task] = 1.0

            model_input_features_clean = ([current_insp_x1_clean] + list(current_instr_oh_clean) +
                                     [insp_r0, insp_r1, insp_ra, insp_f0] +
                                     list(insp_data_memory) + list(current_sp_oh_clean) + [insp_previous_y])
            
            if len(model_input_features_clean) != input_size:
                print(f"WARNUNG: Inkonsistente Feature-Anzahl in Inspektion ({len(model_input_features_clean)}) vs. input_size ({input_size}). Stoppe Inspektion.")
                break
            
            inp_tensor_clean = torch.tensor([model_input_features_clean], dtype=torch.float32).to(DEVICE)

            _ = model_to_train(inp_tensor_clean, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue
            
            model_output_logit = model_to_train(inp_tensor_clean)
            model_output_y = float(torch.sigmoid(model_output_logit[0]).item() > 0.5)

            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            next_insp_r0, next_insp_r1, next_insp_ra, next_insp_f0 = insp_r0, insp_r1, insp_ra, insp_f0
            next_insp_data_memory = list(insp_data_memory)
            correct_target_y_insp = 0.0 
            next_insp_pc_val = insp_program_counter + 1
            mem_addr_insp_clean = int(insp_ra) % data_mem_size_task

            if current_instr_idx_clean_insp == 0: next_insp_r0 = current_insp_x1_clean
            elif current_instr_idx_clean_insp == 1: next_insp_r1 = current_insp_x1_clean
            elif current_instr_idx_clean_insp == 2: next_insp_ra = float(int(current_insp_x1_clean) % data_mem_size_task)
            elif current_instr_idx_clean_insp == 3: next_insp_r0 = insp_r1
            elif current_instr_idx_clean_insp == 4: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1))
            # ADD_R0_R1_R0 ist Index 5 im Dataset, hier entfernt, also ist NOT_R0 jetzt 5
            elif current_instr_idx_clean_insp == 5: next_insp_r0 = float(1 - int(insp_r0)) # NOT_R0
            elif current_instr_idx_clean_insp == 6: next_insp_data_memory[mem_addr_insp_clean] = insp_r0 # STORE_R0_MEM_AT_RA
            elif current_instr_idx_clean_insp == 7: next_insp_r0 = insp_data_memory[mem_addr_insp_clean] # LOAD_R0_FROM_MEM_AT_RA
            elif current_instr_idx_clean_insp == 8: # PUSH_PC_CALL_SUB
                if insp_ret_sp < ret_stack_depth_task: insp_ret_stack[insp_ret_sp] = insp_program_counter + 1; insp_ret_sp += 1
                next_insp_pc_val = SUBROUTINE_1_ADDR_INSP_V2 
            elif current_instr_idx_clean_insp == 9: # RETURN
                if insp_ret_sp > 0: insp_ret_sp -= 1; next_insp_pc_val = insp_ret_stack[insp_ret_sp]
            elif current_instr_idx_clean_insp == 10: next_insp_f0 = 1.0 if int(insp_r0) == 0 else 0.0 # SET_FLAG_IF_R0_ZERO
            elif current_instr_idx_clean_insp == 11: # JUMP_IF_F0_SKIP_N
                if int(insp_f0) == 1: insp_skip_next_n_instr = 2 
            elif current_instr_idx_clean_insp == 12: correct_target_y_insp = next_insp_r0 # OUT_R0
            # Korrekter Index für NO_OP in einem 13-Instruktionen-Set ist 12, wenn von 0 gezählt wird.
            # Die Dataset-Klasse hatte NO_OP als num_instructions-1, was bei 13 Instr. Index 12 ist.
            # ABER die instr_names_sm_v1 Liste hatte 16 Elemente. Das muss konsistent sein.
            # Wir nehmen an, der Datensatz verwendet num_instructions=13 und NO_OP ist 12.
            # Die instr_names_sm_v1 muss gekürzt oder angepasst werden.
            # Für diesen Durchlauf nehme ich an, dass instr_names_sm_v1 mit der Dataset-Definition übereinstimmt (13 Elemente).
            elif current_instr_idx_clean_insp == (num_instructions_task -1) : correct_target_y_insp = insp_previous_y # NO_OP
            
            # Anpassung der instr_names für die Ausgabe, um OutOfBounds zu vermeiden
            instr_display_name = instr_names_sm_v1[current_instr_idx_clean_insp] if current_instr_idx_clean_insp < len(instr_names_sm_v1) else f"UNKNOWN_INSTR({current_instr_idx_clean_insp})"

            dm_str = ",".join(map(str, [int(s) for s in insp_data_memory]))
            print(f"S{sample_count_inspect+1}: x1:{int(current_insp_x1_clean)},I:{instr_display_name}({current_instr_idx_clean_insp}) | R0i:{int(insp_r0)},R1i:{int(insp_r1)},RAi:{int(insp_ra)},F0i:{int(insp_f0)},DM:[{dm_str}],SPi:{insp_ret_sp},y-1i:{int(insp_previous_y)} |Py:{int(model_output_y)},Ty:{int(correct_target_y_insp)},MA:{mean_alpha:.3f}")
            sample_count_inspect += 1

            insp_r0, insp_r1, insp_ra, insp_f0 = next_insp_r0, next_insp_r1, next_insp_ra, next_insp_f0
            insp_data_memory = next_insp_data_memory
            insp_previous_y = correct_target_y_insp
            insp_program_counter = next_insp_pc_val

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")