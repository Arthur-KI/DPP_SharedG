import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 60 # Beibehalten vom letzten erfolgreichen Log
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

# --- Modell-Definitionen (DPPModelBase angepasst für Alpha-Retrieval) ---
class DPPModelBase(nn.Module):
    def __init__(self, dpp_layer, hidden_size_dpp, output_size):
        super(DPPModelBase, self).__init__(); self.dpp_layer1 = dpp_layer
        self.relu1 = nn.ReLU(); self.fc_out = nn.Linear(hidden_size_dpp, output_size)
        self.last_alphas = None

    def forward(self, x, return_alpha_flag=False): # return_alpha_flag wird hier nicht mehr direkt verwendet
        if hasattr(self.dpp_layer1, 'forward') and 'return_alpha' in self.dpp_layer1.forward.__code__.co_varnames:
            out, alphas = self.dpp_layer1(x, return_alpha=True)
            self.last_alphas = alphas # Immer Alphas speichern, wenn möglich
        else:
            out = self.dpp_layer1(x, return_alpha=False)
            self.last_alphas = None
        out = self.relu1(out); out = self.fc_out(out)
        return out

    @torch.no_grad()
    def predict_next_state_and_output(self, model_input_features_clean_tensor):
        self.eval()
        # Der forward-Aufruf setzt self.last_alphas bereits
        output_logit = self(model_input_features_clean_tensor) # return_alpha_flag ist hier nicht nötig
        predicted_y_t = (torch.sigmoid(output_logit) > 0.5).float().item()
        return predicted_y_t

# --- Datengenerierung für "8-Bit CPU V0.1" ---
class CPU_8Bit_V0_1_Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 num_registers=2, register_width=8,
                 num_instructions=12, memory_cells=16,
                 ret_stack_depth=4, pc_bits=8, control_bits_count=8):
        super(CPU_8Bit_V0_1_Dataset, self).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        
        # Parameter direkt als Attribute speichern
        self.NUM_REGISTERS = num_registers
        self.REGISTER_WIDTH = register_width
        self.NUM_INSTRUCTIONS = num_instructions
        self.MEMORY_CELLS = memory_cells
        self.ADDRESS_BITS = int(np.ceil(np.log2(memory_cells))) if memory_cells > 0 else 0
        self.RET_STACK_DEPTH = ret_stack_depth
        self.PC_BITS = pc_bits
        self.MAX_PC_VAL = (2**pc_bits) - 1
        self.CONTROL_BITS_COUNT = control_bits_count

        # Instruktionsdefinitionen
        self.LOAD_R0_IMM = 0; self.LOAD_R1_IMM = 1; self.LOAD_R0_MEM = 2; self.STORE_R0_MEM = 3;
        self.ADD_R0_R1 = 4; self.XOR_R0_R1 = 5; self.NOT_R0 = 6; self.JUMP_IF_ZF = 7;
        self.CALL = 8; self.RET = 9; self.OUT_R0_BIT0 = 10; self.NO_OP = 11;

        self.instr_names = [
            "LDR0IMM", "LDR1IMM", "LDR0MEM", "STR0MEM", "ADDR0R1", "XORR0R1",
            "NOTR0", "JMPZF", "CALL", "RET", "OUTR0B0", "NOOP"
        ]
        assert len(self.instr_names) == self.NUM_INSTRUCTIONS, "Instruction names mismatch"

    def _value_to_bit_array(self, value, width):
        return [float((value >> i) & 1) for i in range(width)]

    def _bit_array_to_value(self, bit_array):
        val = 0
        for i, bit in enumerate(bit_array): val += int(round(bit)) * (2**i)
        return val

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes: one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r = [[0.0] * self.REGISTER_WIDTH for _ in range(self.NUM_REGISTERS)]
            memory = [[0.0] * self.REGISTER_WIDTH for _ in range(self.MEMORY_CELLS)]
            zf = 0.0; cf = 0.0
            ret_stack = [0] * self.RET_STACK_DEPTH
            ret_sp = 0
            previous_y = 0.0
            program_counter = 0

            for t_idx in range(self.seq_len):
                current_instr_idx = np.random.randint(0, self.NUM_INSTRUCTIONS)
                control_data_clean_val = np.random.randint(0, 2**self.CONTROL_BITS_COUNT)
                control_data_bits_clean = self._value_to_bit_array(control_data_clean_val, self.CONTROL_BITS_COUNT)

                if current_instr_idx == self.CALL and ret_sp >= self.RET_STACK_DEPTH: current_instr_idx = self.NO_OP
                if current_instr_idx == self.RET and ret_sp == 0: current_instr_idx = self.NO_OP

                instr_one_hot = self._to_one_hot(current_instr_idx, self.NUM_INSTRUCTIONS)
                sp_one_hot = self._to_one_hot(ret_sp % self.RET_STACK_DEPTH, self.RET_STACK_DEPTH)

                noisy_control_data_bits = [b + np.random.normal(0, self.noise_level) for b in control_data_bits_clean]
                noisy_r = [[bit + np.random.normal(0, self.noise_level) for bit in reg_val] for reg_val in r]
                noisy_memory = [[bit + np.random.normal(0, self.noise_level) for bit in cell_val] for cell_val in memory]
                noisy_zf = zf + np.random.normal(0, self.noise_level)
                noisy_cf = cf + np.random.normal(0, self.noise_level)
                noisy_previous_y = previous_y + np.random.normal(0, self.noise_level)

                input_features = noisy_control_data_bits + list(instr_one_hot)
                for reg_val_bits in noisy_r: input_features.extend(reg_val_bits)
                for mem_val_bits in noisy_memory: input_features.extend(mem_val_bits)
                input_features.extend(list(sp_one_hot))
                input_features.extend([noisy_zf, noisy_cf, noisy_previous_y])
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                next_r = [list(reg_val) for reg_val in r]
                next_memory = [list(cell_val) for cell_val in memory]
                next_zf, next_cf = zf, cf
                target_y = 0.0; pc_jump_target = -1
                operand_val = self._bit_array_to_value(control_data_bits_clean)
                mem_address = 0
                if self.NUM_REGISTERS > 1 and self.MEMORY_CELLS > 0: # Nur wenn R1 und Speicher existieren
                    mem_address = self._bit_array_to_value(r[1]) % self.MEMORY_CELLS


                if current_instr_idx == self.LOAD_R0_IMM: next_r[0] = list(control_data_bits_clean[:self.REGISTER_WIDTH])
                elif current_instr_idx == self.LOAD_R1_IMM:
                    if self.NUM_REGISTERS > 1: next_r[1] = list(control_data_bits_clean[:self.REGISTER_WIDTH])
                elif current_instr_idx == self.LOAD_R0_MEM:
                    if self.MEMORY_CELLS > 0 and self.NUM_REGISTERS > 1: next_r[0] = list(memory[mem_address])
                elif current_instr_idx == self.STORE_R0_MEM:
                    if self.MEMORY_CELLS > 0 and self.NUM_REGISTERS > 1: next_memory[mem_address] = list(r[0])
                elif current_instr_idx == self.ADD_R0_R1:
                    if self.NUM_REGISTERS > 1:
                        val_r0 = self._bit_array_to_value(r[0]); val_r1 = self._bit_array_to_value(r[1])
                        res_val = val_r0 + val_r1
                        next_cf = 1.0 if res_val >= (2**self.REGISTER_WIDTH) else 0.0
                        res_val %= (2**self.REGISTER_WIDTH)
                        next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
                        next_zf = 1.0 if res_val == 0 else 0.0
                elif current_instr_idx == self.XOR_R0_R1:
                    if self.NUM_REGISTERS > 1:
                        res_val = self._bit_array_to_value(r[0]) ^ self._bit_array_to_value(r[1])
                        next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
                        next_zf = 1.0 if res_val == 0 else 0.0; next_cf = 0.0
                elif current_instr_idx == self.NOT_R0:
                    res_val = (~self._bit_array_to_value(r[0])) & ((2**self.REGISTER_WIDTH) - 1)
                    next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
                    next_zf = 1.0 if res_val == 0 else 0.0; next_cf = 0.0
                elif current_instr_idx == self.JUMP_IF_ZF:
                    if int(zf) == 1: pc_jump_target = operand_val % (self.MAX_PC_VAL + 1)
                elif current_instr_idx == self.CALL:
                    if ret_sp < self.RET_STACK_DEPTH: ret_stack[ret_sp] = program_counter + 1; ret_sp +=1
                    pc_jump_target = operand_val % (self.MAX_PC_VAL + 1)
                elif current_instr_idx == self.RET:
                    if ret_sp > 0: ret_sp -= 1; pc_jump_target = ret_stack[ret_sp]
                elif current_instr_idx == self.OUT_R0_BIT0: target_y = next_r[0][0]
                elif current_instr_idx == self.NO_OP: target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r, memory, zf, cf = next_r, next_memory, next_zf, next_cf
                previous_y = target_y
                if pc_jump_target != -1: program_counter = pc_jump_target
                else: program_counter = (program_counter + 1) % (self.MAX_PC_VAL + 1)
    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen (angepasst) ---
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None, scheduler_obj=None):
    model.train()
    history = {'loss': [], 'accuracy': [], 'time_per_epoch': [], 'best_train_acc_epoch': -1, 'best_train_acc': 0.0}
    time_to_target_accuracy = None; epoch_at_target_accuracy = None
    total_training_start_time = time.time()
    best_model_state = None

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0; correct_preds = 0; total_preds = 0; num_batches = 0
        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch: break
            num_batches += 1
            batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP_GLOBAL):
                outputs = model(batch_inputs); loss = criterion(outputs, batch_labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()
            with torch.no_grad(): preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == batch_labels.bool()).sum().item(); total_preds += batch_labels.size(0)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        history['loss'].append(avg_epoch_loss); history['accuracy'].append(accuracy)
        history['time_per_epoch'].append(time.time() - epoch_start_time)

        if accuracy > history['best_train_acc']:
            history['best_train_acc'] = accuracy
            history['best_train_acc_epoch'] = epoch + 1
            best_model_state = model.state_dict()
            print(f"Neues bestes Modell in Epoche {epoch + 1} mit Trainings-Acc: {accuracy:.4f} gespeichert.")

        if scheduler_obj:
            if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler_obj.step(accuracy)
            else: scheduler_obj.step()

        if (accuracy >= target_accuracy and epoch_at_target_accuracy is None) or \
           (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == 0 or epoch == epochs - 1:
            current_total_time = time.time() - total_training_start_time
            lr_print = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] {model_name}, Loss: {avg_epoch_loss:.4f}, Acc: {accuracy:.4f}, LR: {lr_print:.1e}, Total Time: {current_total_time:.2f}s")
        if accuracy >= target_accuracy and epoch_at_target_accuracy is None:
            time_to_target_accuracy = time.time() - total_training_start_time; epoch_at_target_accuracy = epoch + 1
            print(f"--- {model_name} reached target accuracy of {target_accuracy*100:.1f}% at epoch {epoch_at_target_accuracy} in {time_to_target_accuracy:.3f}s ---")
        
        if accuracy >= 0.995 and epoch > epochs * 0.2 :
             print(f"--- {model_name} erreichte hohe Genauigkeit. Training wird frühzeitig in Epoche {epoch + 1} beendet. ---")
             break

    total_training_time = time.time() - total_training_start_time
    print(f"--- Training {model_name} beendet in {total_training_time:.3f}s ---")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Bestes Modell aus Epoche {history['best_train_acc_epoch']} mit Trainings-Acc {history['best_train_acc']:.4f} wurde geladen.")
    return history, total_training_time, epoch_at_target_accuracy, time_to_target_accuracy

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

# --- CPU Simulator für 8-Bit und Programmausführung ---
class CPUSimulator_8Bit:
    def __init__(self, dataset_params_dict):
        self.NUM_REGISTERS = dataset_params_dict["num_registers"]
        self.REGISTER_WIDTH = dataset_params_dict["register_width"]
        self.NUM_INSTRUCTIONS = dataset_params_dict["num_instructions"]
        self.MEMORY_CELLS = dataset_params_dict["memory_cells"]
        self.ADDRESS_BITS = int(np.ceil(np.log2(self.MEMORY_CELLS))) if self.MEMORY_CELLS > 0 else 0
        self.RET_STACK_DEPTH = dataset_params_dict["ret_stack_depth"]
        self.PC_BITS = dataset_params_dict["pc_bits"]
        self.MAX_PC_VAL = (2**self.PC_BITS) - 1
        self.CONTROL_BITS_COUNT = dataset_params_dict["control_bits_count"]

        temp_ds_cpu_config = {
            "num_registers": self.NUM_REGISTERS, "register_width": self.REGISTER_WIDTH,
            "num_instructions": self.NUM_INSTRUCTIONS, "memory_cells": self.MEMORY_CELLS,
            "ret_stack_depth": self.RET_STACK_DEPTH, "pc_bits": self.PC_BITS,
            "control_bits_count": self.CONTROL_BITS_COUNT
        }
        temp_ds = CPU_8Bit_V0_1_Dataset(1, 1, 0.0, **temp_ds_cpu_config)
        self.instr_names = temp_ds.instr_names
        
        self.LOAD_R0_IMM = temp_ds.LOAD_R0_IMM; self.LOAD_R1_IMM = temp_ds.LOAD_R1_IMM;
        self.LOAD_R0_MEM = temp_ds.LOAD_R0_MEM; self.STORE_R0_MEM = temp_ds.STORE_R0_MEM;
        self.ADD_R0_R1 = temp_ds.ADD_R0_R1; self.XOR_R0_R1 = temp_ds.XOR_R0_R1;
        self.NOT_R0 = temp_ds.NOT_R0; self.JUMP_IF_ZF = temp_ds.JUMP_IF_ZF;
        self.CALL = temp_ds.CALL; self.RET = temp_ds.RET;
        self.OUT_R0_BIT0 = temp_ds.OUT_R0_BIT0; self.NO_OP = temp_ds.NO_OP;
        
        self.reset_state()

    def reset_state(self):
        self.r = [[0.0] * self.REGISTER_WIDTH for _ in range(self.NUM_REGISTERS)]
        self.memory = [[0.0] * self.REGISTER_WIDTH for _ in range(self.MEMORY_CELLS)]
        self.zf = 0.0; self.cf = 0.0
        self.ret_stack = [0] * self.RET_STACK_DEPTH
        self.ret_sp = 0
        self.previous_y = 0.0
        self.program_counter = 0
        self.history = []

    def _value_to_bit_array(self, value, width): return [float((value >> i) & 1) for i in range(width)]
    def _bit_array_to_value(self, bit_array):
        val = 0
        for i, bit in enumerate(bit_array): val += int(round(bit)) * (2**i)
        return val
    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes: one_hot[int(value)] = 1.0
        return one_hot

    def step(self, instr_idx_clean, control_data_bits_clean):
        current_state_snapshot = {
            'pc': self.program_counter, 'instr': self.instr_names[instr_idx_clean],
            'ctrl_val': self._bit_array_to_value(control_data_bits_clean),
            'r0': self._bit_array_to_value(self.r[0]), 
            'r1': self._bit_array_to_value(self.r[1]) if self.NUM_REGISTERS > 1 else -1,
            'zf': int(self.zf), 'cf': int(self.cf), 'sp': self.ret_sp,
            'mem0': self._bit_array_to_value(self.memory[0]) if self.MEMORY_CELLS > 0 else -1,
            'y_prev': int(self.previous_y)
        }

        next_r = [list(reg_val) for reg_val in self.r]
        next_memory = [list(cell_val) for cell_val in self.memory]
        next_zf, next_cf = self.zf, self.cf
        target_y = 0.0; pc_jump_target = -1
        operand_val = self._bit_array_to_value(control_data_bits_clean)
        mem_address = 0
        if self.NUM_REGISTERS > 1 and self.MEMORY_CELLS > 0:
             mem_address = self._bit_array_to_value(self.r[1]) % self.MEMORY_CELLS

        if instr_idx_clean == self.LOAD_R0_IMM: next_r[0] = list(control_data_bits_clean[:self.REGISTER_WIDTH])
        elif instr_idx_clean == self.LOAD_R1_IMM:
            if self.NUM_REGISTERS > 1: next_r[1] = list(control_data_bits_clean[:self.REGISTER_WIDTH])
        elif instr_idx_clean == self.LOAD_R0_MEM:
            if self.MEMORY_CELLS > 0 and self.NUM_REGISTERS > 1: next_r[0] = list(self.memory[mem_address])
        elif instr_idx_clean == self.STORE_R0_MEM:
            if self.MEMORY_CELLS > 0 and self.NUM_REGISTERS > 1: next_memory[mem_address] = list(self.r[0])
        elif instr_idx_clean == self.ADD_R0_R1:
            if self.NUM_REGISTERS > 1:
                val_r0 = self._bit_array_to_value(self.r[0]); val_r1 = self._bit_array_to_value(self.r[1])
                res_val = val_r0 + val_r1
                next_cf = 1.0 if res_val >= (2**self.REGISTER_WIDTH) else 0.0
                res_val %= (2**self.REGISTER_WIDTH)
                next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
                next_zf = 1.0 if res_val == 0 else 0.0
        elif instr_idx_clean == self.XOR_R0_R1:
            if self.NUM_REGISTERS > 1:
                res_val = self._bit_array_to_value(self.r[0]) ^ self._bit_array_to_value(self.r[1])
                next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
                next_zf = 1.0 if res_val == 0 else 0.0; next_cf = 0.0
        elif instr_idx_clean == self.NOT_R0:
            res_val = (~self._bit_array_to_value(self.r[0])) & ((2**self.REGISTER_WIDTH) - 1)
            next_r[0] = self._value_to_bit_array(res_val, self.REGISTER_WIDTH)
            next_zf = 1.0 if res_val == 0 else 0.0; next_cf = 0.0
        elif instr_idx_clean == self.JUMP_IF_ZF:
            if int(self.zf) == 1: pc_jump_target = operand_val % (self.MAX_PC_VAL + 1)
        elif instr_idx_clean == self.CALL:
            if self.ret_sp < self.RET_STACK_DEPTH: self.ret_stack[self.ret_sp] = self.program_counter + 1; self.ret_sp +=1
            pc_jump_target = operand_val % (self.MAX_PC_VAL + 1)
        elif instr_idx_clean == self.RET:
            if self.ret_sp > 0: self.ret_sp -= 1; pc_jump_target = self.ret_stack[self.ret_sp]
        elif instr_idx_clean == self.OUT_R0_BIT0: target_y = next_r[0][0]
        elif instr_idx_clean == self.NO_OP: target_y = self.previous_y

        self.r, self.memory, self.zf, self.cf = next_r, next_memory, next_zf, next_cf
        self.previous_y = target_y
        if pc_jump_target != -1: self.program_counter = pc_jump_target
        else: self.program_counter = (self.program_counter + 1) % (self.MAX_PC_VAL + 1)

        current_state_snapshot['y_out_expected'] = int(target_y)
        self.history.append(current_state_snapshot)
        return target_y

    def get_model_input_features(self, instr_idx_clean, control_data_bits_clean):
        instr_one_hot = self._to_one_hot(instr_idx_clean, self.NUM_INSTRUCTIONS)
        sp_one_hot = self._to_one_hot(self.ret_sp % self.RET_STACK_DEPTH, self.RET_STACK_DEPTH)
        input_f = list(control_data_bits_clean) + list(instr_one_hot)
        for reg_val_bits in self.r: input_f.extend(reg_val_bits)
        for mem_val_bits in self.memory: input_f.extend(mem_val_bits)
        input_f.extend(list(sp_one_hot))
        input_f.extend([self.zf, self.cf, self.previous_y])
        return torch.tensor([input_f], dtype=torch.float32).to(DEVICE)

def execute_program_on_8bit_model_and_sim(model, program_dict, simulator, max_steps=100):
    model.eval()
    simulator.reset_state()
    print("\n--- 8-Bit CPU Programmausführung Start ---")
    header = f"{'SimPC':>5} | {'Instr':<7} | {'CtrlV':>5} | {'R0':>3} | {'R1':>3} | {'ZF':>2} | {'CF':>2} | {'Mem[0..1]':<7} | SPkTop | {'y_prev':>6} | {'Exp_Y':>5} | {'Mod_Y':>5} | {'Alpha':>5}"
    print(header)
    print("-" * (len(header) + 5))

    correct_y_predictions = 0; executed_steps = 0
    for step_count in range(max_steps):
        current_pc_sim = simulator.program_counter
        if current_pc_sim not in program_dict:
            print(f"PC {current_pc_sim} nicht im Programm. Ende.")
            break
        instr_idx, control_val = program_dict[current_pc_sim]
        control_bits = simulator._value_to_bit_array(control_val, simulator.CONTROL_BITS_COUNT)

        model_input_tensor = simulator.get_model_input_features(instr_idx, control_bits)
        predicted_y = model.predict_next_state_and_output(model_input_tensor)
        mean_alpha = np.mean(model.last_alphas.cpu().numpy()) if model.last_alphas is not None else -1.0

        expected_y = simulator.step(instr_idx, control_bits)
        state_before_exec = simulator.history[-1]

        if int(predicted_y) == int(expected_y): correct_y_predictions += 1
        executed_steps += 1

        stack_top_val = simulator.ret_stack[state_before_exec['sp']-1] if state_before_exec['sp'] > 0 else -1
        mem_display_vals = []
        for i in range(min(2, simulator.MEMORY_CELLS)): # Zeige bis zu 2 Speicherzellen
            mem_display_vals.append(simulator._bit_array_to_value(simulator.memory[i]))
        mem_display_str = str(mem_display_vals)


        print(f"{state_before_exec['pc']:>5} | {state_before_exec['instr']:<7} | {state_before_exec['ctrl_val']:>5} | "
              f"{state_before_exec['r0']:>3} | {state_before_exec['r1']:>3} | {state_before_exec['zf']:>2} | {state_before_exec['cf']:>2} | "
              f"{mem_display_str:<7} | {state_before_exec['sp']:>1}{stack_top_val:>3} | "
              f"{state_before_exec['y_prev']:>6} | {int(expected_y):>5} | {int(predicted_y):>5} | {mean_alpha:.3f}")

        if instr_idx == simulator.NO_OP and current_pc_sim == simulator.program_counter :
             if program_dict.get(current_pc_sim +1) is None:
                print("Programmende durch NOOP und keine weiteren Instruktionen erreicht.")
                break

    accuracy = (correct_y_predictions / executed_steps) * 100 if executed_steps > 0 else 0.0
    print("-" * (len(header) + 5))
    print(f"8-Bit Programmausführung beendet nach {executed_steps} Schritten. Genauigkeit: {accuracy:.2f}% ({correct_y_predictions}/{executed_steps})")
    return accuracy

# --- Hauptteil ---
if __name__ == "__main__":
    set_seed(SEED)

    cpu_config = {
        "num_registers": 2, "register_width": 8, "num_instructions": 12,
        "memory_cells": 4,
        "ret_stack_depth": 4, "pc_bits": 8, "control_bits_count": 8
    }

    input_size = cpu_config["control_bits_count"] + \
                 cpu_config["num_instructions"] + \
                 (cpu_config["num_registers"] * cpu_config["register_width"]) + \
                 (cpu_config["memory_cells"] * cpu_config["register_width"]) + \
                 cpu_config["ret_stack_depth"] + \
                 1 + 1 + 1 # ZF, CF, prev_y

    output_size = 1
    learning_rate = 0.0005
    batch_size = 1024
    epochs = 5 # Reduziert von 15 für den allerersten Testlauf dieses komplexen Codes
    noise_level_data = 0.001
    target_accuracy_threshold = 0.60

    dpp_units = 128
    shared_g_dim_config = input_size // 4 if input_size // 4 > 0 else 1 # Sicherstellen, dass > 0

    print(f"Task: 8-Bit CPU V0.1 Simulation (Testlauf {SEED})")
    print(f"Input size: {input_size}, Memory Cells: {cpu_config['memory_cells']}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    num_train_sequences_per_epoch = 50000
    num_test_sequences_per_epoch = 5000
    seq_len_task = 30

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps/epoch: {train_steps_per_epoch}, Test steps/epoch: {test_steps_per_epoch}")

    train_dataset = CPU_8Bit_V0_1_Dataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, **cpu_config)
    test_dataset = CPU_8Bit_V0_1_Dataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, **cpu_config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}")

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-7)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.3, min_lr=1e-8, verbose=True) # Kürzere Patience für Test

    print(f"\n--- Training DPP SharedG für 8-Bit CPU V0.1 ---")
    history, _, _, _ = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_CPU_8Bit_V0.1_Run1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (8-Bit CPU, Zufallsdaten) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_CPU_8Bit_V0.1_Run1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_CPU_8Bit_V0.1_Run1 - Parameter: {param_count}")
    print(f"  Final Training Acc (letzte Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Acc: {final_acc:.4f}, Final Test Loss: {final_loss:.4f}")

    # --- Testprogramm für 8-Bit CPU ---
    cpu_sim_8bit = CPUSimulator_8Bit(cpu_config)
    LDR0IMM = cpu_sim_8bit.LOAD_R0_IMM; LDR1IMM = cpu_sim_8bit.LOAD_R1_IMM
    LDR0MEM = cpu_sim_8bit.LOAD_R0_MEM; STR0MEM = cpu_sim_8bit.STORE_R0_MEM
    ADDR0R1 = cpu_sim_8bit.ADD_R0_R1; XORR0R1 = cpu_sim_8bit.XOR_R0_R1
    NOTR0 = cpu_sim_8bit.NOT_R0; JMPZF = cpu_sim_8bit.JUMP_IF_ZF
    CALL = cpu_sim_8bit.CALL; RET = cpu_sim_8bit.RET
    OUTR0B0 = cpu_sim_8bit.OUT_R0_BIT0; NOOP = cpu_sim_8bit.NO_OP
    SUB_ADDR = 20 

    program_8bit_final_test = {
        0: (LDR0IMM, 5), 1: (LDR1IMM, 2), 2: (STR0MEM, 0), 3: (LDR0IMM, 10),
        4: (LDR1IMM, 3), 5: (ADDR0R1, 0), 6: (OUTR0B0, 0), 7: (LDR1IMM, 2),
        8: (LDR0MEM, 0), 9: (OUTR0B0, 0), 10: (NOTR0, 0), 11: (OUTR0B0, 0),
        12: (LDR0IMM, 0), 13: (LDR1IMM, 0), 14: (ADDR0R1, 0), 15: (JMPZF, 18),
        16: (LDR0IMM, 99), 17: (OUTR0B0, 0), 18: (CALL, SUB_ADDR),
        19: (OUTR0B0, 0),
        SUB_ADDR -1 : (NOOP, 0), # PC 19 - Sicherstellen, dass es existiert
        SUB_ADDR + 0: (LDR0IMM, 7), SUB_ADDR + 1: (NOTR0, 0), SUB_ADDR + 2: (RET, 0),
        SUB_ADDR + 3: (NOOP, 0) # Ende der Subroutine, um sicherzustellen, dass PC danach definiert ist
    }
    execute_program_on_8bit_model_and_sim(model_to_train, program_8bit_final_test, cpu_sim_8bit, max_steps=50)