import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 57 # Neuer Seed für diesen FPLA-Test
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

# --- Datengenerierung für "FPLA V1" ---
class FPLA_V1_Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 num_instructions=16, data_mem_size=4, ret_stack_depth=4):
        super(FPLA_V1_Dataset, self).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions # Sollte mit der Länge der instr_names übereinstimmen
        self.data_mem_size = data_mem_size
        self.ret_stack_depth = ret_stack_depth

        # Instruktions-Indizes (16 Instruktionen, 0-15)
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.LOAD_RA_X = 2        # RA (Adressregister) mit x1_t laden (Index 0..data_mem_size-1)
        self.MOVE_R0_R1 = 3
        self.ALU_XOR_R0_R1 = 4    # R0 <- R0 XOR R1
        self.ALU_AND_R0_R1 = 5    # R0 <- R0 AND R1
        self.ALU_NOT_R0 = 6       # R0 <- NOT R0
        self.STORE_R0_AT_RA = 7   # DataMem[RA] <- R0
        self.LOAD_R0_FROM_RA = 8  # R0 <- DataMem[RA]
        self.SET_F0_IF_R0_ZERO = 9
        self.JUMP_IF_F0_SKIP_N = 10 # N ist fest (z.B. 3)
        self.CALL_SUB_FIXED = 11  # Ruft eine feste Subroutine-Adresse auf
        self.RETURN_FROM_SUB = 12
        self.OUT_R0 = 13
        self.OUT_MEM_AT_RA = 14
        self.NO_OP = 15
        
        self.JUMP_SKIP_AMOUNT = 3 # Für JUMP_IF_F0_SKIP_N

        self.instr_names = [
            "LOAD_R0_X", "LOAD_R1_X", "LOAD_RA_X", "MOVE_R0_R1", "ALU_XOR_R0_R1",
            "ALU_AND_R0_R1", "ALU_NOT_R0", "STORE_R0_AT_RA", "LOAD_R0_FROM_RA",
            "SET_F0_IF_R0_ZERO", "JUMP_IF_F0_SKIP_N", "CALL_SUB_FIXED", "RETURN_FROM_SUB",
            "OUT_R0", "OUT_MEM_AT_RA", "NO_OP"
        ]
        assert len(self.instr_names) == self.num_instructions, "Mismatch in instruction names and num_instructions"

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes:
            one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        SUBROUTINE_START_ADDR = self.seq_len + 10 # Fiktive Adresse

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
                elif program_counter == SUBROUTINE_START_ADDR:
                    if idx_in_sequence >= self.seq_len: break
                    current_instr_idx_clean = self.ALU_NOT_R0 
                elif program_counter == SUBROUTINE_START_ADDR + 1:
                    if idx_in_sequence >= self.seq_len: break
                    current_instr_idx_clean = self.MOVE_R0_R1 
                elif program_counter == SUBROUTINE_START_ADDR + 2: 
                    if idx_in_sequence >= self.seq_len: break
                    current_instr_idx_clean = self.RETURN_FROM_SUB
                else: 
                    current_instr_idx_clean = np.random.randint(0, self.num_instructions)
                    if current_instr_idx_clean == self.CALL_SUB_FIXED and ret_sp >= self.ret_stack_depth:
                        current_instr_idx_clean = self.NO_OP
                    if current_instr_idx_clean == self.RETURN_FROM_SUB and ret_sp == 0:
                        current_instr_idx_clean = self.NO_OP
                
                instruction_one_hot = self._to_one_hot(current_instr_idx_clean, self.num_instructions)
                ra_one_hot = self._to_one_hot(ra, self.data_mem_size) # RA als One-Hot
                sp_one_hot = self._to_one_hot(ret_sp % self.ret_stack_depth, self.ret_stack_depth)

                x1_t_noisy = current_x1_bit_clean + np.random.normal(0, self.noise_level)
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                # ra_one_hot nicht verrauschen
                f0_tm1_noisy = f0 + np.random.normal(0, self.noise_level)
                data_mem_noisy = [m_val + np.random.normal(0, self.noise_level) for m_val in data_memory]
                # sp_one_hot nicht verrauschen
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)
                
                input_features = ([x1_t_noisy] + list(instruction_one_hot) +
                                  [r0_tm1_noisy, r1_tm1_noisy] + list(ra_one_hot) + [f0_tm1_noisy] +
                                  data_mem_noisy + list(sp_one_hot) + [y_tm1_noisy])
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                next_r0, next_r1, next_ra, next_f0 = r0, r1, ra, f0
                next_data_memory = list(data_memory)
                target_y = 0.0
                next_pc_val = program_counter + 1
                mem_addr_clean = int(ra) 

                if current_instr_idx_clean == self.LOAD_R0_X: next_r0 = current_x1_bit_clean
                elif current_instr_idx_clean == self.LOAD_R1_X: next_r1 = current_x1_bit_clean
                elif current_instr_idx_clean == self.LOAD_RA_X: 
                    next_ra = float(int(current_x1_bit_clean) % self.data_mem_size)
                elif current_instr_idx_clean == self.MOVE_R0_R1: next_r0 = r1
                elif current_instr_idx_clean == self.ALU_XOR_R0_R1: next_r0 = float(int(r0) ^ int(r1))
                elif current_instr_idx_clean == self.ALU_AND_R0_R1: next_r0 = float(int(r0) & int(r1))
                elif current_instr_idx_clean == self.ALU_NOT_R0: next_r0 = float(1 - int(r0))
                elif current_instr_idx_clean == self.STORE_R0_AT_RA: next_data_memory[mem_addr_clean] = r0
                elif current_instr_idx_clean == self.LOAD_R0_FROM_RA: next_r0 = data_memory[mem_addr_clean]
                elif current_instr_idx_clean == self.SET_F0_IF_R0_ZERO:
                    next_f0 = 1.0 if int(r0) == 0 else 0.0
                elif current_instr_idx_clean == self.JUMP_IF_F0_SKIP_N:
                    if int(f0) == 1: skip_next_n_instr = self.JUMP_SKIP_AMOUNT
                elif current_instr_idx_clean == self.CALL_SUB_FIXED:
                    if ret_sp < self.ret_stack_depth:
                        ret_stack[ret_sp] = program_counter + 1
                        ret_sp = (ret_sp + 1) 
                    next_pc_val = SUBROUTINE_START_ADDR
                elif current_instr_idx_clean == self.RETURN_FROM_SUB:
                    if ret_sp > 0:
                        ret_sp = ret_sp - 1
                        next_pc_val = ret_stack[ret_sp]
                elif current_instr_idx_clean == self.OUT_R0: target_y = next_r0
                elif current_instr_idx_clean == self.OUT_MEM_AT_RA: target_y = next_data_memory[mem_addr_clean]
                elif current_instr_idx_clean == self.NO_OP: target_y = previous_y
                
                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r0, r1, ra, f0 = next_r0, next_r1, next_ra, next_f0
                data_memory = next_data_memory
                previous_y = target_y
                program_counter = next_pc_val
                idx_in_sequence +=1
                
    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen (unverändert) ---
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None, scheduler_obj=None):
    model.train()
    history={'loss':[],'accuracy':[],'time_per_epoch':[]}
    time_to_target_accuracy=None; epoch_at_target_accuracy=None
    total_training_start_time=time.time()
    consecutive_target_epochs = 0 
    early_stop_threshold = 10 

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
        
        if accuracy >= 0.9999: 
            consecutive_target_epochs += 1
            if consecutive_target_epochs >= early_stop_threshold:
                print(f"--- {model_name} reached stable 100% accuracy for {early_stop_threshold} epochs. Stopping training early at epoch {epoch+1}. ---")
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
    
    num_instructions_task = 16 # Gemäß FPLA_V1_Dataset Definition
    data_mem_size_task = 4
    ret_stack_depth_task = 4
    
    # Input: x1(1) + instr_oh(16) + R0(1) + R1(1) + RA_oh(4) + F0(1) + DataMem(4) + SP_oh(4) + y_tm1(1) = 33
    input_size = 1 + num_instructions_task + 1 + 1 + data_mem_size_task + 1 + data_mem_size_task + ret_stack_depth_task + 1

    output_size = 1
    learning_rate = 0.0008 
    batch_size = 2048 
    epochs = 5 # Lange Trainingszeit für diese extrem komplexe Aufgabe
    noise_level_data = 0.001 
    target_accuracy_threshold = 0.60 # Sehr konservatives Ziel

    dpp_units = 96
    shared_g_dim_config = 16 # input_size (~33) / 2 = ~16

    print(f"Task: FPLA V1 (Registers, Stack, Mem, Flag, Jumps, Calls)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    num_train_sequences_per_epoch = 75000 
    num_test_sequences_per_epoch = 15000
    seq_len_task = 70

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = FPLA_V1_Dataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task)
    test_dataset = FPLA_V1_Dataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, data_mem_size_task, ret_stack_depth_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte ca. 8593 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-8)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50, factor=0.2, min_lr=1e-8, verbose=True) 

    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für FPLA V1 ---")
    
    # Wenn Sie einen vorherigen Lauf abbrechen und neu starten, möchten Sie vielleicht die Epochenzahl reduzieren
    # Hier mit vollen 500 Epochen, aber Ihr manueller Abbruch ist verständlich.
    # epochs_this_run = 5 # Für einen schnellen Test wie in Ihrem Log
    epochs_this_run = epochs 
    
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs_this_run, model_name="DPP_FPLA_V1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (FPLA V1) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_FPLA_V1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_FPLA_V1 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs_this_run} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (FPLA V1) ---")
    model_to_train.eval()
    max_inspect_alpha_fpla_v1 = 30 

    inspector_dataset_fpla_v1 = FPLA_V1_Dataset(1, max_inspect_alpha_fpla_v1 + seq_len_task, 0.0, num_instructions_task, data_mem_size_task, ret_stack_depth_task)
    
    instr_names_fpla_v1 = inspector_dataset_fpla_v1.instr_names


    print(f"Format Input: [x1, instr_oh({num_instructions_task}), R0,R1,RA_oh({data_mem_size_task}),F0, DataMem({data_mem_size_task}), SP_oh({ret_stack_depth_task}), y-1]")
    print("----------------------------------------------------------------------------------------------------")
    
    with torch.no_grad():
        insp_r0, insp_r1, insp_ra, insp_f0 = 0.0, 0.0, 0.0, 0.0
        insp_data_memory = [0.0] * data_mem_size_task
        insp_ret_stack = [0] * ret_stack_depth_task
        insp_ret_sp = 0
        insp_previous_y = 0.0
        insp_program_counter = 0
        insp_skip_next_n_instr = 0
        SUBROUTINE_START_ADDR_INSP_V3 = seq_len_task + 10


        for sample_count_inspect in range(max_inspect_alpha_fpla_v1):
            current_insp_x1_clean = float(random.randint(0,1))
            current_instr_idx_clean_insp = 0
            
            if insp_skip_next_n_instr > 0:
                current_instr_idx_clean_insp = inspector_dataset_fpla_v1.NO_OP
                insp_skip_next_n_instr -= 1
            elif insp_program_counter == SUBROUTINE_START_ADDR_INSP_V3 : 
                current_instr_idx_clean_insp = inspector_dataset_fpla_v1.ALU_NOT_R0
            elif insp_program_counter == SUBROUTINE_START_ADDR_INSP_V3 + 1: 
                current_instr_idx_clean_insp = inspector_dataset_fpla_v1.MOVE_R0_R1
            elif insp_program_counter == SUBROUTINE_START_ADDR_INSP_V3 + 2: 
                current_instr_idx_clean_insp = inspector_dataset_fpla_v1.RETURN_FROM_SUB
            else:
                current_instr_idx_clean_insp = random.randint(0, num_instructions_task -1)
                if current_instr_idx_clean_insp == inspector_dataset_fpla_v1.CALL_SUB_FIXED and insp_ret_sp >= ret_stack_depth_task: 
                    current_instr_idx_clean_insp = inspector_dataset_fpla_v1.NO_OP
                if current_instr_idx_clean_insp == inspector_dataset_fpla_v1.RETURN_FROM_SUB and insp_ret_sp == 0: 
                    current_instr_idx_clean_insp = inspector_dataset_fpla_v1.NO_OP

            current_instr_oh_clean = inspector_dataset_fpla_v1._to_one_hot(current_instr_idx_clean_insp, num_instructions_task)
            current_ra_oh_clean = inspector_dataset_fpla_v1._to_one_hot(insp_ra, data_mem_size_task)
            current_sp_oh_clean = inspector_dataset_fpla_v1._to_one_hot(insp_ret_sp % ret_stack_depth_task, ret_stack_depth_task)

            model_input_features_clean = ([current_insp_x1_clean] + list(current_instr_oh_clean) +
                                     [insp_r0, insp_r1] + list(current_ra_oh_clean) + [insp_f0] +
                                     list(insp_data_memory) + list(current_sp_oh_clean) + [insp_previous_y])
            
            if len(model_input_features_clean) != input_size:
                print(f"WARNUNG: Feature-Anzahl in Inspektion ({len(model_input_features_clean)}) vs. input_size ({input_size}). Überspringe Sample.")
                continue # Überspringe dieses Sample
            
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
            mem_addr_insp_clean = int(insp_ra)

            ds = inspector_dataset_fpla_v1 
            if current_instr_idx_clean_insp == ds.LOAD_R0_X: next_insp_r0 = current_insp_x1_clean
            elif current_instr_idx_clean_insp == ds.LOAD_R1_X: next_insp_r1 = current_insp_x1_clean
            elif current_instr_idx_clean_insp == ds.LOAD_RA_X: next_insp_ra = float(int(current_insp_x1_clean) % ds.data_mem_size)
            elif current_instr_idx_clean_insp == ds.MOVE_R0_R1: next_insp_r0 = insp_r1
            elif current_instr_idx_clean_insp == ds.ALU_XOR_R0_R1: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1))
            elif current_instr_idx_clean_insp == ds.ALU_AND_R0_R1: next_insp_r0 = float(int(insp_r0) & int(insp_r1))
            elif current_instr_idx_clean_insp == ds.ALU_NOT_R0: next_insp_r0 = float(1 - int(insp_r0))
            elif current_instr_idx_clean_insp == ds.STORE_R0_AT_RA: next_insp_data_memory[mem_addr_insp_clean] = insp_r0
            elif current_instr_idx_clean_insp == ds.LOAD_R0_FROM_RA: next_insp_r0 = insp_data_memory[mem_addr_insp_clean]
            elif current_instr_idx_clean_insp == ds.SET_F0_IF_R0_ZERO: next_insp_f0 = 1.0 if int(insp_r0) == 0 else 0.0
            elif current_instr_idx_clean_insp == ds.JUMP_IF_F0_SKIP_N:
                if int(insp_f0) == 1: insp_skip_next_n_instr = ds.JUMP_SKIP_AMOUNT 
            elif current_instr_idx_clean_insp == ds.CALL_SUB_FIXED:
                if insp_ret_sp < ds.ret_stack_depth: insp_ret_stack[insp_ret_sp] = insp_program_counter + 1; insp_ret_sp = (insp_ret_sp + 1)
                next_insp_pc_val = SUBROUTINE_START_ADDR_INSP_V3 
            elif current_instr_idx_clean_insp == ds.RETURN_FROM_SUB:
                if insp_ret_sp > 0: insp_ret_sp = insp_ret_sp - 1; next_insp_pc_val = insp_ret_stack[insp_ret_sp]
            elif current_instr_idx_clean_insp == ds.OUT_R0: correct_target_y_insp = next_insp_r0 
            elif current_instr_idx_clean_insp == ds.OUT_MEM_AT_RA: correct_target_y_insp = next_insp_data_memory[mem_addr_insp_clean]
            elif current_instr_idx_clean_insp == ds.NO_OP: correct_target_y_insp = insp_previous_y
            
            instr_display_name = ds.instr_names[current_instr_idx_clean_insp]
            dm_str = ",".join(map(str, [int(s) for s in insp_data_memory]))
            ra_oh_str = "".join(map(str, [int(s) for s in current_ra_oh_clean])) 
            sp_oh_str = "".join(map(str, [int(s) for s in current_sp_oh_clean]))

            print(f"S{sample_count_inspect+1}: x1:{int(current_insp_x1_clean)},I:{instr_display_name}({current_instr_idx_clean_insp})|R0i:{int(insp_r0)},R1i:{int(insp_r1)},RAi_oh:{ra_oh_str}(val {int(insp_ra)}),F0i:{int(insp_f0)},DM:[{dm_str}],SPi_oh:{sp_oh_str}(val {insp_ret_sp}),y-1i:{int(insp_previous_y)}|Py:{int(model_output_y)},Ty:{int(correct_target_y_insp)},MA:{mean_alpha:.3f}")
            
            insp_r0, insp_r1, insp_ra, insp_f0 = next_insp_r0, next_insp_r1, next_insp_ra, next_insp_f0
            insp_data_memory = next_insp_data_memory
            insp_previous_y = correct_target_y_insp
            insp_program_counter = next_insp_pc_val
            sample_count_inspect += 1

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")