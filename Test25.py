import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 61 
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

    def forward(self, x, return_alpha_flag=False): 
        if hasattr(self.dpp_layer1, 'forward') and 'return_alpha' in self.dpp_layer1.forward.__code__.co_varnames:
            out, alphas = self.dpp_layer1(x, return_alpha=True)
            self.last_alphas = alphas 
        else:
            try:
                out, alphas = self.dpp_layer1(x, return_alpha=True)
                self.last_alphas = alphas
            except TypeError: 
                out = self.dpp_layer1(x) 
                self.last_alphas = None
        out = self.relu1(out); out = self.fc_out(out)
        return out

    @torch.no_grad()
    def predict_output(self, model_input_features_clean_tensor):
        self.eval()
        output_logit = self(model_input_features_clean_tensor) 
        predicted_y_t = (torch.sigmoid(output_logit) > 0.5).float().item()
        return predicted_y_t

# --- Datengenerierung für "Simplified 32-Bit CPU V0.1" ---
class CPU_32Bit_Simplified_Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 register_width=32, num_gp_registers=2, 
                 pc_width=16, 
                 num_instructions=8, 
                 memory_cells=4,     
                 visible_memory_window_cells=2, 
                 control_bits_for_operand=32,
                 ret_stack_depth=0, 
                 **kwargs 
                 ):
        super(CPU_32Bit_Simplified_Dataset, self).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level

        self.REG_WIDTH = register_width
        self.NUM_GP_REGS = num_gp_registers
        self.PC_WIDTH = pc_width
        self.MAX_PC_VAL = (2**pc_width) - 1
        self.NUM_INSTRUCTIONS = num_instructions
        self.MEMORY_CELLS = memory_cells
        self.VISIBLE_MEM_WINDOW_CELLS = min(memory_cells, visible_memory_window_cells)
        self.CONTROL_BITS_COUNT = control_bits_for_operand

        self.FLAG_Z = 0; self.FLAG_C = 1;
        self.NUM_FLAGS = 2

        self.LOAD_R0_IMM = 0; self.LOAD_R1_IMM = 1; self.ADD_R0_R1   = 2; self.XOR_R0_R1   = 3;
        self.STORE_R0_ADDR = 4; self.LOAD_R0_ADDR  = 5; self.OUT_R0_LSB  = 6; self.NO_OP       = 7;
        assert self.NUM_INSTRUCTIONS == 8

        self.instr_names = [
            "LDR0IMM", "LDR1IMM", "ADDR0R1", "XORR0R1",
            "STR0ADDR", "LDR0ADDR", "OUTR0LSB", "NOOP"
        ]
        assert len(self.instr_names) == self.NUM_INSTRUCTIONS, "Instruction names mismatch"

    def _val_to_bits(self, val, width): return [float((val >> i) & 1) for i in range(width)]
    def _bits_to_val(self, bits):
        val = 0
        for i, bit in enumerate(bits): val += int(round(bit)) * (2**i)
        return val

    def _to_one_hot(self, val, classes):
        oh = np.zeros(classes, dtype=np.float32)
        if 0 <= val < classes: oh[int(val)] = 1.0
        return oh

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r = [[0.0] * self.REG_WIDTH for _ in range(self.NUM_GP_REGS)] 
            memory = [[0.0] * self.REG_WIDTH for _ in range(self.MEMORY_CELLS)]
            flags = [0.0] * self.NUM_FLAGS 
            pc_val = 0
            prev_y = 0.0

            for t in range(self.seq_len):
                instr_idx = np.random.randint(0, self.NUM_INSTRUCTIONS)
                control_val = np.random.randint(0, 2**self.CONTROL_BITS_COUNT, dtype=np.int64)
                control_bits = self._val_to_bits(control_val, self.CONTROL_BITS_COUNT)

                instr_oh = self._to_one_hot(instr_idx, self.NUM_INSTRUCTIONS)
                pc_bits_arr = self._val_to_bits(pc_val, self.PC_WIDTH)
                
                mem_view_flat = []
                for i in range(self.VISIBLE_MEM_WINDOW_CELLS):
                    mem_view_flat.extend(memory[i % self.MEMORY_CELLS])

                current_input_features = list(control_bits) + list(instr_oh)
                for reg_val_bits in r: current_input_features.extend(reg_val_bits)
                current_input_features.extend(list(pc_bits_arr))
                current_input_features.extend(list(flags))
                current_input_features.extend(mem_view_flat)
                current_input_features.append(prev_y)
                
                noisy_input_features = []
                idx = 0
                noisy_input_features.extend([b + np.random.normal(0, self.noise_level) for b in current_input_features[idx:idx+self.CONTROL_BITS_COUNT]])
                idx += self.CONTROL_BITS_COUNT
                noisy_input_features.extend(current_input_features[idx:idx+self.NUM_INSTRUCTIONS])
                idx += self.NUM_INSTRUCTIONS
                for _ in range(self.NUM_GP_REGS):
                    noisy_input_features.extend([b + np.random.normal(0, self.noise_level) for b in current_input_features[idx:idx+self.REG_WIDTH]])
                    idx += self.REG_WIDTH
                noisy_input_features.extend([b + np.random.normal(0, self.noise_level) for b in current_input_features[idx:idx+self.PC_WIDTH]])
                idx += self.PC_WIDTH
                noisy_input_features.extend([b + np.random.normal(0, self.noise_level) for b in current_input_features[idx:idx+self.NUM_FLAGS]])
                idx += self.NUM_FLAGS
                noisy_input_features.extend([b + np.random.normal(0, self.noise_level) for b in current_input_features[idx:idx+len(mem_view_flat)]])
                idx += len(mem_view_flat)
                noisy_input_features.append(current_input_features[idx] + np.random.normal(0, self.noise_level))
                input_vector = torch.tensor(noisy_input_features, dtype=torch.float32)

                next_r = [list(reg_val) for reg_val in r]
                next_memory = [list(cell_val) for cell_val in memory]
                next_flags = list(flags)
                target_y = 0.0
                next_pc_val = (pc_val + 1) % (self.MAX_PC_VAL + 1)
                operand32 = self._bits_to_val(control_bits)

                if instr_idx == self.LOAD_R0_IMM: next_r[0] = list(control_bits)
                elif instr_idx == self.LOAD_R1_IMM: 
                    if self.NUM_GP_REGS > 1: next_r[1] = list(control_bits)
                elif instr_idx == self.ADD_R0_R1:
                    if self.NUM_GP_REGS > 1:
                        val0 = self._bits_to_val(r[0]); val1 = self._bits_to_val(r[1])
                        res = val0 + val1
                        next_flags[self.FLAG_C] = 1.0 if res >= (2**self.REG_WIDTH) else 0.0
                        res &= ((2**self.REG_WIDTH) - 1)
                        next_r[0] = self._val_to_bits(res, self.REG_WIDTH)
                        next_flags[self.FLAG_Z] = 1.0 if res == 0 else 0.0
                elif instr_idx == self.XOR_R0_R1:
                    if self.NUM_GP_REGS > 1:
                        val0 = self._bits_to_val(r[0]); val1 = self._bits_to_val(r[1])
                        res = val0 ^ val1
                        next_r[0] = self._val_to_bits(res, self.REG_WIDTH)
                        next_flags[self.FLAG_Z] = 1.0 if res == 0 else 0.0
                        next_flags[self.FLAG_C] = 0.0 
                elif instr_idx == self.STORE_R0_ADDR:
                    addr = operand32 % self.MEMORY_CELLS
                    if 0 <= addr < self.MEMORY_CELLS: next_memory[addr] = list(r[0])
                elif instr_idx == self.LOAD_R0_ADDR:
                    addr = operand32 % self.MEMORY_CELLS
                    if 0 <= addr < self.MEMORY_CELLS: next_r[0] = list(memory[addr])
                elif instr_idx == self.OUT_R0_LSB: target_y = next_r[0][0]
                elif instr_idx == self.NO_OP: target_y = prev_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r, memory, flags = next_r, next_memory, next_flags
                pc_val = next_pc_val
                prev_y = target_y
    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen ---
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
           (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == 0 or epoch == epochs - 1: # Geändert auf //10 für längere Trainings
            current_total_time = time.time() - total_training_start_time
            lr_print = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] {model_name}, Loss: {avg_epoch_loss:.4f}, Acc: {accuracy:.4f}, LR: {lr_print:.1e}, Total Time: {current_total_time:.2f}s")
        if accuracy >= target_accuracy and epoch_at_target_accuracy is None:
            time_to_target_accuracy = time.time() - total_training_start_time; epoch_at_target_accuracy = epoch + 1
            print(f"--- {model_name} reached target accuracy of {target_accuracy*100:.1f}% at epoch {epoch_at_target_accuracy} in {time_to_target_accuracy:.3f}s ---")
        
        if accuracy >= 0.995 and epoch > epochs * 0.1 : # Geändertes Early Stopping Kriterium
             print(f"--- {model_name} erreichte hohe Genauigkeit über mehrere Schritte. Training wird frühzeitig in Epoche {epoch + 1} beendet. ---")
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

# --- Hauptteil ---
if __name__ == "__main__":
    set_seed(SEED)

    cpu_32bit_simplified_config = {
        "register_width": 32,
        "num_gp_registers": 2,
        "pc_width": 16,
        "num_instructions": 8,
        "memory_cells": 4,
        "visible_memory_window_cells": 2,
        "control_bits_for_operand": 32,
        "ret_stack_depth": 0 
    }

    input_size = (cpu_32bit_simplified_config["control_bits_for_operand"] +
                  cpu_32bit_simplified_config["num_instructions"] +
                  (cpu_32bit_simplified_config["num_gp_registers"] * cpu_32bit_simplified_config["register_width"]) +
                  cpu_32bit_simplified_config["pc_width"] +
                  2 +  # Vereinfachte Flags (Z, C)
                  (cpu_32bit_simplified_config["visible_memory_window_cells"] * cpu_32bit_simplified_config["register_width"]) +
                  1)   # prev_y

    output_size = 1
    # Angepasste Trainingsparameter
    learning_rate = 5e-5 
    batch_size = 128       # Beibehalten wie im Log
    epochs = 100           # Erhöht auf 100
    noise_level_data = 0.0001
    target_accuracy_threshold = 0.95 # Ambitionierteres Ziel

    # Modellparameter aus dem Log beibehalten
    dpp_units = 96 
    shared_g_dim_config = 18

    print(f"Task: Simplified 32-Bit CPU V0.1 Simulation (Testlauf {SEED} - Mehr Epochen/Daten)")
    print(f"Input size: {input_size}, Visible Memory Cells: {cpu_32bit_simplified_config['visible_memory_window_cells']}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Angepasste Datenmengen
    num_train_sequences_per_epoch = 20000 # Erhöht
    num_test_sequences_per_epoch = 2000  # Erhöht
    seq_len_task = 20                    # Erhöht

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps/epoch: {train_steps_per_epoch}, Test steps/epoch: {test_steps_per_epoch}")

    dataset_args_template = {
        "register_width": cpu_32bit_simplified_config["register_width"],
        "num_gp_registers": cpu_32bit_simplified_config["num_gp_registers"],
        "pc_width": cpu_32bit_simplified_config["pc_width"],
        "num_instructions": cpu_32bit_simplified_config["num_instructions"],
        "memory_cells": cpu_32bit_simplified_config["memory_cells"],
        "visible_memory_window_cells": cpu_32bit_simplified_config["visible_memory_window_cells"],
        "control_bits_for_operand": cpu_32bit_simplified_config["control_bits_for_operand"],
        "ret_stack_depth": cpu_32bit_simplified_config["ret_stack_depth"]
    }

    train_dataset_args = {**dataset_args_template,
                          "num_sequences_per_epoch": num_train_sequences_per_epoch,
                          "seq_len": seq_len_task,
                          "noise_level": noise_level_data}
    train_dataset = CPU_32Bit_Simplified_Dataset(**train_dataset_args)

    test_dataset_args = {**dataset_args_template,
                         "num_sequences_per_epoch": num_test_sequences_per_epoch,
                         "seq_len": seq_len_task,
                         "noise_level": noise_level_data}
    test_dataset = CPU_32Bit_Simplified_Dataset(**test_dataset_args)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte weiterhin ~41k sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-8)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.7, min_lr=1e-9, verbose=True) # Angepasste Patience

    print(f"\n--- Training DPP SharedG für Simplified 32-Bit CPU ---")
    history, _, _, _ = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_CPU_32Bit_Simp_Run2", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation (Simplified 32-Bit CPU, Zufallsdaten) ---")
    if epochs > 0 and len(history['accuracy']) > 0 : 
        final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_CPU_32Bit_Simp_Run2", steps_per_epoch=test_steps_per_epoch)
        print(f"DPP_CPU_32Bit_Simp_Run2 - Parameter: {param_count}")
        print(f"  Final Training Acc (letzte Epoche): {history['accuracy'][-1]:.4f}")
        print(f"  Final Test Acc: {final_acc:.4f}, Final Test Loss: {final_loss:.4f}")
    else:
        print("Kein Training durchgeführt oder keine History vorhanden für finale Evaluation.")

    print("\n--- Programmausführung für Simplified 32-Bit CPU (Platzhalter) ---")
    print("Eine detaillierte Programmausführung würde eine entsprechende Simulator-Klasse und ein Testprogramm erfordern.")