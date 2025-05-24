import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 58 # Neuer Seed für diesen "CPU-Kern"-Test
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

# --- Datengenerierung für "CPU-Kern V0.1" ---
class CPU_Core_V0_1_Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 num_instructions=18, data_mem_size=8, ret_stack_depth=4, pc_bits=6):
        super(CPU_Core_V0_1_Dataset, self).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.data_mem_size = data_mem_size # 2^3 = 8
        self.ret_stack_depth = ret_stack_depth # 2^2 = 4
        self.pc_bits = pc_bits # Max PC value = 2^pc_bits - 1
        self.max_pc_val = (2**pc_bits) -1


        # Instruktions-Indizes (18 Instruktionen)
        self.LOAD_R0_X = 0      # Rx from x1_t, c1_t, c2_t (00=R0, 01=R1, 10=R2, 11=R3)
        self.LOAD_R1_X = 1      # Diese sind vereinfacht zu direkten LOADs für R0/R1
        self.LOAD_R2_X = 2      # (R2 wird nicht direkt geladen, kann durch andere Ops befüllt werden)
        self.LOAD_AR_X = 3      # AR from x1_t,c1_t,c2_t (3 bits for 0..7)
        self.MOVE_R0_R1 = 4     # R0 <- R1
        self.ALU_XOR_R1R2_R0 = 5  # R0 <- R1 XOR R2, sets ZF, EQF
        self.ALU_AND_R1R2_R0 = 6  # R0 <- R1 AND R2, sets ZF, EQF
        self.ALU_OR_R1R2_R0 = 7   # R0 <- R1 OR R2, sets ZF, EQF
        self.ALU_NOT_R1_R0 = 8    # R0 <- NOT R1, sets ZF
        self.STORE_R0_MEM_AR = 9  # DataMem[AR] <- R0
        self.LOAD_R0_MEM_AR = 10  # R0 <- DataMem[AR]
        self.INC_AR = 11          # AR <- (AR+1) % DataMemSize
        self.DEC_AR = 12          # AR <- (AR-1) % DataMemSize
        self.JUMP_IF_ZF_ADDR = 13 # PC <- addr (from x1,c1-c5) if ZF=1
        self.CALL_ADDR = 14       # Push PC+1, PC <- addr (from x1,c1-c5)
        self.RETURN = 15
        self.OUT_R0 = 16
        self.NO_OP = 17

        self.instr_names = [
            "LDR0X", "LDR1X", "LDR2X", "LDARX", "MOVR0R1", "XOR", "AND", "OR", "NOT",
            "STOMEM","LDMEM", "INCAR", "DECAR", "JMPZF", "CALL", "RET", "OUTR0", "NOOP"
        ]
        assert len(self.instr_names) == self.num_instructions

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes: one_hot[int(value)] = 1.0
        return one_hot
    
    def _decode_val_from_control_bits(self, bits_list):
        val = 0
        for i, bit in enumerate(bits_list):
            val += int(round(bit)) * (2**i)
        return val

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r = [0.0] * 4 # R0, R1, R2, R3
            ar = 0.0      # Adressregister Wert (0..data_mem_size-1)
            zf, eqf = 0.0, 0.0 # Flags
            data_memory = [0.0] * self.data_mem_size
            ret_stack = [0] * self.ret_stack_depth # Speichert PC-Werte
            ret_sp = 0 
            previous_y = 0.0
            program_counter = 0

            idx_in_sequence = 0
            while idx_in_sequence < self.seq_len:
                # Kontrollbits/Datenbits für die aktuelle Instruktion generieren
                # x1_t, c1_t, c2_t, c3_t, c4_t, c5_t (6 Bits)
                control_data_bits_clean = [float(np.random.randint(0,2)) for _ in range(6)]
                
                current_instr_idx_clean = np.random.randint(0, self.num_instructions)

                # Logik um ungültige CALL/RETURN zu vermeiden
                if current_instr_idx_clean == self.CALL_ADDR and ret_sp >= self.ret_stack_depth:
                    current_instr_idx_clean = self.NO_OP
                if current_instr_idx_clean == self.RETURN and ret_sp == 0:
                    current_instr_idx_clean = self.NO_OP

                instruction_one_hot = self._to_one_hot(current_instr_idx_clean, self.num_instructions)
                ar_one_hot = self._to_one_hot(ar, self.data_mem_size)
                sp_one_hot = self._to_one_hot(ret_sp % self.ret_stack_depth, self.ret_stack_depth)

                # Inputs für das Modell (mit Rauschen)
                control_data_bits_noisy = [b + np.random.normal(0, self.noise_level) for b in control_data_bits_clean]
                r_noisy = [reg + np.random.normal(0, self.noise_level) for reg in r]
                # ar_one_hot nicht verrauschen
                zf_noisy = zf + np.random.normal(0, self.noise_level)
                eqf_noisy = eqf + np.random.normal(0, self.noise_level)
                data_mem_noisy = [m_val + np.random.normal(0, self.noise_level) for m_val in data_memory]
                # sp_one_hot nicht verrauschen
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)
                
                input_features = control_data_bits_noisy + list(instruction_one_hot) + \
                                 r_noisy + list(ar_one_hot) + [zf_noisy, eqf_noisy] + \
                                 data_mem_noisy + list(sp_one_hot) + [y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Nächste Zustände und Ziel-y (basierend auf sauberen Werten)
                next_r = list(r)
                next_ar = ar
                next_zf, next_eqf = zf, eqf
                next_data_memory = list(data_memory)
                target_y = 0.0
                pc_jump_target = -1 # Für JUMP/CALL/RETURN

                # Registerauswahl (Rx aus c1,c2; Ry aus c3,c4 - falls benötigt)
                # Für LOAD_RX_VAL: Rx ist durch c1,c2; val ist control_data_bits_clean[0] (x1_t)
                rx_idx_load = self._decode_val_from_control_bits(control_data_bits_clean[1:3]) % 4 # R0-R3
                
                # Für ALU: R1, R2 sind feste Operanden, R0 ist Ziel
                # Für JUMP_ADDR, CALL_ADDR: Adresse aus 6 control_data_bits
                addr_from_bits = self._decode_val_from_control_bits(control_data_bits_clean) % (self.max_pc_val +1)


                if current_instr_idx_clean == self.LOAD_R0_X: next_r[0] = control_data_bits_clean[0]
                elif current_instr_idx_clean == self.LOAD_R1_X: next_r[1] = control_data_bits_clean[0]
                elif current_instr_idx_clean == self.LOAD_R2_X: next_r[2] = control_data_bits_clean[0] # R2 laden
                elif current_instr_idx_clean == self.LOAD_AR_X: 
                    next_ar = float(self._decode_val_from_control_bits(control_data_bits_clean[0:3]) % self.data_mem_size) # AR aus 3 Bits laden
                elif current_instr_idx_clean == self.MOVE_R0_R1: next_r[0] = r[1]
                elif current_instr_idx_clean == self.ALU_XOR_R1R2_R0: 
                    next_r[0] = float(int(r[1]) ^ int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_AND_R1R2_R0: 
                    next_r[0] = float(int(r[1]) & int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_OR_R1R2_R0:
                    next_r[0] = float(int(r[1]) | int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_NOT_R1_R0: 
                    next_r[0] = float(1 - int(r[1])); next_zf = 1.0 if next_r[0] == 0 else 0.0
                elif current_instr_idx_clean == self.STORE_R0_MEM_AR: next_data_memory[int(ar)] = r[0]
                elif current_instr_idx_clean == self.LOAD_R0_MEM_AR: next_r[0] = data_memory[int(ar)]
                elif current_instr_idx_clean == self.INC_AR: next_ar = (ar + 1) % self.data_mem_size
                elif current_instr_idx_clean == self.DEC_AR: next_ar = (ar - 1 + self.data_mem_size) % self.data_mem_size
                elif current_instr_idx_clean == self.JUMP_IF_ZF_ADDR:
                    if int(zf) == 1: pc_jump_target = addr_from_bits
                elif current_instr_idx_clean == self.CALL_ADDR:
                    if ret_sp < self.ret_stack_depth: ret_stack[ret_sp] = program_counter + 1; ret_sp = (ret_sp + 1)
                    pc_jump_target = addr_from_bits
                elif current_instr_idx_clean == self.RETURN:
                    if ret_sp > 0: ret_sp = ret_sp - 1; pc_jump_target = ret_stack[ret_sp]
                elif current_instr_idx_clean == self.OUT_R0: target_y = next_r[0]
                # OUT_MEM_AT_AR entfernt für dieses Set, um bei 16 Instr. zu bleiben, NO_OP ist 15
                elif current_instr_idx_clean == self.NO_OP: target_y = previous_y
                
                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r, ar, zf, eqf = next_r, next_ar, next_zf, next_eqf
                data_memory = next_data_memory
                previous_y = target_y
                if pc_jump_target != -1: program_counter = pc_jump_target
                else: program_counter += 1
                program_counter %= (self.max_pc_val +1) # Wrap around PC
                idx_in_sequence +=1
                
    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen (unverändert) ---
# (Identisch zu vorher, hier gekürzt für Übersichtlichkeit)
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None, scheduler_obj=None):
    model.train()
    history={'loss':[],'accuracy':[],'time_per_epoch':[]}
    time_to_target_accuracy=None; epoch_at_target_accuracy=None
    total_training_start_time=time.time()
    consecutive_target_epochs = 0 
    early_stop_threshold = 20 # Mehr Geduld für diese komplexe Aufgabe

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
        
        # Early stopping, wenn eine hohe Genauigkeit stabil erreicht wird
        if accuracy >= 0.995: # Hohe Schwelle für diese komplexe Aufgabe
            consecutive_target_epochs += 1
            if consecutive_target_epochs >= early_stop_threshold:
                print(f"--- {model_name} reached stable high accuracy for {early_stop_threshold} epochs. Stopping training early at epoch {epoch+1}. ---")
                break
        else:
            consecutive_target_epochs = 0
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
    
    num_instructions_task = 18 # Gemäß Definition im Dataset
    data_mem_size_task = 8   # 8 Speicherzellen
    ret_stack_depth_task = 4 # 4 Ebenen für Return-Stack
    pc_bits_task = 6         # PC kann Werte 0-63 annehmen
    
    # Input: control_data_bits(6) + instr_oh(18) + R0-3(4) + AR_oh(8) + ZF,EQF(2) + DataMem(8) + SP_oh(4) + y_tm1(1) = 51
    input_size = 6 + num_instructions_task + 4 + data_mem_size_task + 2 + data_mem_size_task + ret_stack_depth_task + 1

    output_size = 1
    learning_rate = 0.0005 # Niedrigere LR für diese sehr komplexe Aufgabe
    batch_size = 4096      # Sehr große Batch Size
    epochs = 10          # Extrem viele Epochen, aber mit Early Stopping
    noise_level_data = 0.001 
    target_accuracy_threshold = 0.50 # Sehr, sehr konservatives Ziel, um überhaupt ein Signal zu sehen

    dpp_units = 128
    shared_g_dim_config = 25 # input_size (~51) / 2

    print(f"Task: CPU-Kern V0.1")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    num_train_sequences_per_epoch = 100000 # Viele Programme
    num_test_sequences_per_epoch = 20000
    seq_len_task = 100 # Sehr lange Programme (maximale PC-Länge ist 2^pc_bits_task)

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = CPU_Core_V0_1_Dataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task, pc_bits_task)
    test_dataset = CPU_Core_V0_1_Dataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task, pc_bits_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte ~18069 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-8)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=75, factor=0.2, min_lr=1e-9, verbose=True) 

    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für CPU-Kern V0.1 ---")
    
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_CPU_Core_V0_1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (CPU-Kern V0.1) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_CPU_Core_V0_1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_CPU_Core_V0_1 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion (sehr rudimentär für diese Komplexität)
    print("\n--- Alpha-Inspektion für einige Test-Samples (CPU-Kern V0.1) ---")
    model_to_train.eval()
    max_inspect_alpha_cpu_v01 = 30

    inspector_dataset_cpu_v01 = CPU_Core_V0_1_Dataset(1, max_inspect_alpha_cpu_v01 + seq_len_task, 0.0, num_instructions_task, data_mem_size_task, ret_stack_depth_task, pc_bits_task)
    
    instr_names_cpu_v01 = inspector_dataset_cpu_v01.instr_names

    print(f"Format Input: [ctrl_data(6), instr_oh({num_instructions_task}), R0-3(4), AR_oh({data_mem_size_task}), Flags(2), DataMem({data_mem_size_task}), SP_oh({ret_stack_depth_task}), y-1(1)]")
    print("----------------------------------------------------------------------------------------------------")
    
    with torch.no_grad():
        # Manuelle Simulation für Inspektion, um saubere Zustände zu haben
        insp_r = [0.0] * 4
        insp_ar = 0.0
        insp_zf, insp_eqf = 0.0, 0.0
        insp_data_memory = [0.0] * data_mem_size_task
        insp_ret_stack = [0] * ret_stack_depth_task
        insp_ret_sp = 0
        insp_previous_y = 0.0
        insp_program_counter = 0
        insp_skip_next_n_instr = 0
        SUBROUTINE_START_ADDR_INSP_V4 = seq_len_task + 10


        for sample_count_inspect in range(max_inspect_alpha_cpu_v01):
            current_insp_control_data_clean = [float(np.random.randint(0,2)) for _ in range(6)]
            current_instr_idx_clean_insp = 0
            
            if insp_skip_next_n_instr > 0:
                current_instr_idx_clean_insp = inspector_dataset_cpu_v01.NO_OP
                insp_skip_next_n_instr -= 1
            # Hier einfache Subroutinen-Logik für Inspektion, kann erweitert werden
            elif insp_program_counter == SUBROUTINE_START_ADDR_INSP_V4 : 
                current_instr_idx_clean_insp = inspector_dataset_cpu_v01.ALU_NOT_R1_R0 # R0 <- NOT R1
            elif insp_program_counter == SUBROUTINE_START_ADDR_INSP_V4 + 1: 
                current_instr_idx_clean_insp = inspector_dataset_cpu_v01.RETURN
            else:
                current_instr_idx_clean_insp = random.randint(0, num_instructions_task -1)
                if current_instr_idx_clean_insp == inspector_dataset_cpu_v01.CALL_ADDR and insp_ret_sp >= ret_stack_depth_task: 
                    current_instr_idx_clean_insp = inspector_dataset_cpu_v01.NO_OP
                if current_instr_idx_clean_insp == inspector_dataset_cpu_v01.RETURN and insp_ret_sp == 0: 
                    current_instr_idx_clean_insp = inspector_dataset_cpu_v01.NO_OP

            current_instr_oh_clean = inspector_dataset_cpu_v01._to_one_hot(current_instr_idx_clean_insp, num_instructions_task)
            current_ar_oh_clean = inspector_dataset_cpu_v01._to_one_hot(insp_ar, data_mem_size_task)
            current_sp_oh_clean = inspector_dataset_cpu_v01._to_one_hot(insp_ret_sp % ret_stack_depth_task, ret_stack_depth_task)

            model_input_features_clean = (current_insp_control_data_clean + list(current_instr_oh_clean) +
                                     insp_r + list(current_ar_oh_clean) + [insp_zf, insp_eqf] +
                                     list(insp_data_memory) + list(current_sp_oh_clean) + [insp_previous_y])
            
            if len(model_input_features_clean) != input_size:
                print(f"WARNUNG: Feature-Anzahl in Inspektion ({len(model_input_features_clean)}) vs. input_size ({input_size}). Überspringe.")
                continue
            
            inp_tensor_clean = torch.tensor([model_input_features_clean], dtype=torch.float32).to(DEVICE)

            _ = model_to_train(inp_tensor_clean, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue
            
            model_output_logit = model_to_train(inp_tensor_clean)
            model_output_y = float(torch.sigmoid(model_output_logit[0]).item() > 0.5)

            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            # Korrekte Zustände für die Anzeige
            next_insp_r = list(insp_r)
            next_insp_ar = insp_ar
            next_insp_zf, next_insp_eqf = insp_zf, insp_eqf
            next_insp_data_memory = list(insp_data_memory)
            correct_target_y_insp = 0.0 
            next_insp_pc_val = insp_program_counter + 1
            
            # Registerauswahl für LOAD_RX_VAL
            rx_idx_load_insp = inspector_dataset_cpu_v01._decode_val_from_control_bits(current_insp_control_data_clean[1:3]) % 4
            val_for_load_insp = current_insp_control_data_clean[0]

            # Adresse für JUMP/CALL
            addr_for_jump_call_insp = inspector_dataset_cpu_v01._decode_val_from_control_bits(current_insp_control_data_clean) % (inspector_dataset_cpu_v01.max_pc_val +1)


            ds_insp = inspector_dataset_cpu_v01 
            if current_instr_idx_clean_insp == ds_insp.LOAD_R0_X: next_insp_r[0] = val_for_load_insp # Angenommen Rx ist durch Opcode, val ist x1
            elif current_instr_idx_clean_insp == ds_insp.LOAD_R1_X: next_insp_r[1] = val_for_load_insp
            elif current_instr_idx_clean_insp == ds_insp.LOAD_R2_X: next_insp_r[2] = val_for_load_insp # R2 für Allgemeinheit
            elif current_instr_idx_clean_insp == ds_insp.LOAD_AR_X: 
                next_insp_ar = float(inspector_dataset_cpu_v01._decode_val_from_control_bits(current_insp_control_data_clean[0:3]) % ds_insp.data_mem_size)
            elif current_instr_idx_clean_insp == ds_insp.MOVE_R0_R1: next_insp_r[0] = insp_r[1] # R0 <- R1
            elif current_instr_idx_clean_insp == ds_insp.ALU_XOR_R1R2_R0: 
                res = float(int(insp_r[1]) ^ int(insp_r[2])); next_insp_r[0] = res; next_insp_zf = 1.0 if res == 0 else 0.0; next_insp_eqf = 1.0 if int(insp_r[1])==int(insp_r[2]) else 0.0
            elif current_instr_idx_clean_insp == ds_insp.ALU_AND_R1R2_R0:
                res = float(int(insp_r[1]) & int(insp_r[2])); next_insp_r[0] = res; next_insp_zf = 1.0 if res == 0 else 0.0; next_insp_eqf = 1.0 if int(insp_r[1])==int(insp_r[2]) else 0.0
            elif current_instr_idx_clean_insp == ds_insp.ALU_OR_R1R2_R0:
                res = float(int(insp_r[1]) | int(insp_r[2])); next_insp_r[0] = res; next_insp_zf = 1.0 if res == 0 else 0.0; next_insp_eqf = 1.0 if int(insp_r[1])==int(insp_r[2]) else 0.0
            elif current_instr_idx_clean_insp == ds_insp.ALU_NOT_R1_R0:
                res = float(1-int(insp_r[1])); next_insp_r[0] = res; next_insp_zf = 1.0 if res == 0 else 0.0
            elif current_instr_idx_clean_insp == ds_insp.STORE_R0_MEM_AR: next_insp_data_memory[int(insp_ar)] = insp_r[0]
            elif current_instr_idx_clean_insp == ds_insp.LOAD_R0_MEM_AR: next_insp_r[0] = insp_data_memory[int(insp_ar)]
            elif current_instr_idx_clean_insp == ds_insp.INC_AR: next_insp_ar = (insp_ar + 1) % ds_insp.data_mem_size
            elif current_instr_idx_clean_insp == ds_insp.DEC_AR: next_insp_ar = (insp_ar - 1 + ds_insp.data_mem_size) % ds_insp.data_mem_size
            elif current_instr_idx_clean_insp == ds_insp.JUMP_IF_ZF_ADDR:
                if int(insp_zf) == 1: next_insp_pc_val = addr_for_jump_call_insp
            elif current_instr_idx_clean_insp == ds_insp.CALL_ADDR:
                if insp_ret_sp < ds_insp.ret_stack_depth: insp_ret_stack[insp_ret_sp] = insp_program_counter + 1; insp_ret_sp +=1
                next_insp_pc_val = addr_for_jump_call_insp
            elif current_instr_idx_clean_insp == ds_insp.RETURN:
                if insp_ret_sp > 0: insp_ret_sp -= 1; next_insp_pc_val = insp_ret_stack[insp_ret_sp]
            elif current_instr_idx_clean_insp == ds_insp.OUT_R0: correct_target_y_insp = next_insp_r[0] 
            # OUT_MEM_AT_AR fehlt in dieser 16-Instruktionen Version, NO_OP ist 15
            elif current_instr_idx_clean_insp == ds_insp.NO_OP: correct_target_y_insp = insp_previous_y
            
            instr_display_name = ds_insp.instr_names[current_instr_idx_clean_insp]
            regs_str = ",".join(map(str, [int(s) for s in insp_r]))
            ar_oh_str = "".join(map(str, [int(s) for s in current_ar_oh_clean]))
            dm_str = ",".join(map(str, [int(s) for s in insp_data_memory]))
            sp_oh_str = "".join(map(str, [int(s) for s in current_sp_oh_clean]))

            print(f"S{sample_count_inspect+1}: x_ctrl:{''.join(map(str, [int(b) for b in current_insp_control_data_clean]))},I:{instr_display_name}({current_instr_idx_clean_insp})|Regs_i:[{regs_str}],ARi_oh:{ar_oh_str}(v {int(insp_ar)}),ZF_i:{int(insp_zf)},EQF_i:{int(insp_eqf)},DMi:[{dm_str}],SPi_oh:{sp_oh_str}(v {insp_ret_sp-1 if current_instr_idx_clean_insp == ds_insp.RETURN and insp_ret_sp > 0 else insp_ret_sp}),y-1i:{int(insp_previous_y)}|Py:{int(model_output_y)},Ty:{int(correct_target_y_insp)},MA:{mean_alpha:.3f}")
            
            insp_r, insp_ar, insp_zf, insp_eqf = next_insp_r, next_insp_ar, next_insp_zf, next_insp_eqf
            insp_data_memory = next_insp_data_memory
            insp_previous_y = correct_target_y_insp
            insp_program_counter = next_insp_pc_val % (ds_insp.max_pc_val + 1) # PC wrap around
            sample_count_inspect += 1

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")