import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 58 # Beibehalten von Test20.py
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

# --- DPPLayer_SharedG Klasse (unverändert aus Test20.py) ---
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

# --- Modell-Definitionen (DPPModelBase bleibt gleich aus Test20.py) ---
class DPPModelBase(nn.Module):
    def __init__(self, dpp_layer, hidden_size_dpp, output_size):
        super(DPPModelBase, self).__init__(); self.dpp_layer1 = dpp_layer
        self.relu1 = nn.ReLU(); self.fc_out = nn.Linear(hidden_size_dpp, output_size)
        self.last_alphas = None
        self.predicted_states_for_program = [] # Für die Programmausführung

    def forward(self, x, return_alpha_flag=False):
        if return_alpha_flag and hasattr(self.dpp_layer1, 'forward') and \
           'return_alpha' in self.dpp_layer1.forward.__code__.co_varnames:
            out, alphas = self.dpp_layer1(x, return_alpha=True)
            self.last_alphas = alphas
        else:
            out = self.dpp_layer1(x, return_alpha=False)
        out = self.relu1(out); out = self.fc_out(out)
        return out

    @torch.no_grad()
    def predict_next_state_and_output(self, model_input_features_clean_tensor):
        """
        Diese Methode ist neu und wird für die schrittweise Programmausführung benötigt.
        Sie nimmt den sauberen Input-Tensor, macht eine Vorhersage für y_t
        und gibt auch die "gedachten" nächsten Zustände zurück,
        die das Modell intern im DPPLayer hätte, wenn es die Logik perfekt gelernt hat.
        Für die tatsächliche Demonstration müssen wir y_t vorhersagen.
        Die internen Zustände (Register etc.) werden hier nicht direkt vom Modell vorhergesagt,
        sondern wir simulieren sie basierend auf der Instruktion, um den nächsten Input zu bauen.
        """
        self.eval()
        # Das Modell sagt nur y_t vorher.
        # Die Logik zur Aktualisierung von Registern, Speicher etc. ist in der CPU-Logik (Simulator)
        # oder muss vom Modell implizit durch die Alpha-Werte und Pfade gelernt werden, um y_t korrekt vorherzusagen.
        # Hier geben wir nur den Output y_t zurück.
        output_logit = self(model_input_features_clean_tensor, return_alpha_flag=False)
        predicted_y_t = (torch.sigmoid(output_logit) > 0.5).float().item()
        return predicted_y_t


# --- Datengenerierung für "CPU-Kern V0.1" (unverändert aus Test20.py) ---
class CPU_Core_V0_1_Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0,
                 num_instructions=18, data_mem_size=8, ret_stack_depth=4, pc_bits=6):
        super(CPU_Core_V0_1_Dataset, self).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.data_mem_size = data_mem_size
        self.ret_stack_depth = ret_stack_depth
        self.pc_bits = pc_bits
        self.max_pc_val = (2**pc_bits) -1

        self.LOAD_R0_X = 0; self.LOAD_R1_X = 1; self.LOAD_R2_X = 2; self.LOAD_AR_X = 3
        self.MOVE_R0_R1 = 4; self.ALU_XOR_R1R2_R0 = 5; self.ALU_AND_R1R2_R0 = 6
        self.ALU_OR_R1R2_R0 = 7; self.ALU_NOT_R1_R0 = 8; self.STORE_R0_MEM_AR = 9
        self.LOAD_R0_MEM_AR = 10; self.INC_AR = 11; self.DEC_AR = 12
        self.JUMP_IF_ZF_ADDR = 13; self.CALL_ADDR = 14; self.RETURN = 15
        self.OUT_R0 = 16; self.NO_OP = 17

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
            r = [0.0] * 4
            ar = 0.0
            zf, eqf = 0.0, 0.0
            data_memory = [0.0] * self.data_mem_size
            ret_stack = [0] * self.ret_stack_depth
            ret_sp = 0
            previous_y = 0.0
            program_counter = 0
            idx_in_sequence = 0
            while idx_in_sequence < self.seq_len:
                control_data_bits_clean = [float(np.random.randint(0,2)) for _ in range(6)]
                current_instr_idx_clean = np.random.randint(0, self.num_instructions)
                if current_instr_idx_clean == self.CALL_ADDR and ret_sp >= self.ret_stack_depth: current_instr_idx_clean = self.NO_OP
                if current_instr_idx_clean == self.RETURN and ret_sp == 0: current_instr_idx_clean = self.NO_OP

                instruction_one_hot = self._to_one_hot(current_instr_idx_clean, self.num_instructions)
                ar_one_hot = self._to_one_hot(ar, self.data_mem_size)
                sp_one_hot = self._to_one_hot(ret_sp % self.ret_stack_depth, self.ret_stack_depth)

                control_data_bits_noisy = [b + np.random.normal(0, self.noise_level) for b in control_data_bits_clean]
                r_noisy = [reg_val + np.random.normal(0, self.noise_level) for reg_val in r]
                zf_noisy = zf + np.random.normal(0, self.noise_level)
                eqf_noisy = eqf + np.random.normal(0, self.noise_level)
                data_mem_noisy = [m_val + np.random.normal(0, self.noise_level) for m_val in data_memory]
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                input_features = control_data_bits_noisy + list(instruction_one_hot) + \
                                 r_noisy + list(ar_one_hot) + [zf_noisy, eqf_noisy] + \
                                 data_mem_noisy + list(sp_one_hot) + [y_tm1_noisy]
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                next_r = list(r); next_ar = ar; next_zf, next_eqf = zf, eqf
                next_data_memory = list(data_memory); target_y = 0.0; pc_jump_target = -1
                addr_from_bits = self._decode_val_from_control_bits(control_data_bits_clean) % (self.max_pc_val +1)

                if current_instr_idx_clean == self.LOAD_R0_X: next_r[0] = control_data_bits_clean[0]
                elif current_instr_idx_clean == self.LOAD_R1_X: next_r[1] = control_data_bits_clean[0]
                elif current_instr_idx_clean == self.LOAD_R2_X: next_r[2] = control_data_bits_clean[0]
                elif current_instr_idx_clean == self.LOAD_AR_X: next_ar = float(self._decode_val_from_control_bits(control_data_bits_clean[0:3]) % self.data_mem_size)
                elif current_instr_idx_clean == self.MOVE_R0_R1: next_r[0] = r[1]
                elif current_instr_idx_clean == self.ALU_XOR_R1R2_R0: next_r[0] = float(int(r[1]) ^ int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_AND_R1R2_R0: next_r[0] = float(int(r[1]) & int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_OR_R1R2_R0: next_r[0] = float(int(r[1]) | int(r[2])); next_zf = 1.0 if next_r[0] == 0 else 0.0; next_eqf = 1.0 if int(r[1])==int(r[2]) else 0.0
                elif current_instr_idx_clean == self.ALU_NOT_R1_R0: next_r[0] = float(1 - int(r[1])); next_zf = 1.0 if next_r[0] == 0 else 0.0
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
                elif current_instr_idx_clean == self.NO_OP: target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r, ar, zf, eqf = next_r, next_ar, next_zf, next_eqf
                data_memory = next_data_memory; previous_y = target_y
                if pc_jump_target != -1: program_counter = pc_jump_target
                else: program_counter += 1
                program_counter %= (self.max_pc_val +1); idx_in_sequence +=1

    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen (unverändert aus Test20.py) ---
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None, scheduler_obj=None):
    model.train()
    history={'loss':[],'accuracy':[],'time_per_epoch':[], 'best_val_acc_epoch': -1, 'best_val_acc': 0.0} # Hinzugefügt, um das beste Modell zu speichern
    time_to_target_accuracy=None; epoch_at_target_accuracy=None
    total_training_start_time=time.time()
    consecutive_target_epochs = 0
    early_stop_threshold = 20
    best_model_state = None # Variable, um das beste Modell zu speichern

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

        if accuracy > history['best_val_acc']: # Speichere das beste Modell basierend auf Trainingsgenauigkeit
            history['best_val_acc'] = accuracy
            history['best_val_acc_epoch'] = epoch + 1
            best_model_state = model.state_dict()
            print(f"Neues bestes Modell in Epoche {epoch+1} mit Trainings-Acc: {accuracy:.4f} gespeichert.")


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

        if accuracy >= 0.999: # Angepasste Schwelle für komplexere Aufgabe
            consecutive_target_epochs += 1
            if consecutive_target_epochs >= early_stop_threshold:
                print(f"--- {model_name} reached stable high accuracy for {early_stop_threshold} epochs. Stopping training early at epoch {epoch+1}. ---")
                break
        else:
            consecutive_target_epochs = 0
    total_training_time=time.time()-total_training_start_time
    print(f"--- Training {model_name} finished in {total_training_time:.3f}s ---")
    if best_model_state:
        model.load_state_dict(best_model_state) # Lade das beste Modell zurück
        print(f"Bestes Modell aus Epoche {history['best_val_acc_epoch']} mit Trainings-Acc {history['best_val_acc']:.4f} wurde geladen.")
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

# --- CPU Simulator und Programmausführung ---
class CPUSimulator:
    def __init__(self, num_instructions=18, data_mem_size=8, ret_stack_depth=4, pc_bits=6):
        self.cpu_params = {
            "num_instructions": num_instructions,
            "data_mem_size": data_mem_size,
            "ret_stack_depth": ret_stack_depth,
            "pc_bits": pc_bits,
            "max_pc_val": (2**pc_bits) - 1
        }
        # Instruktions-Indizes kopieren für den Simulator
        ds = CPU_Core_V0_1_Dataset(1,1) # Dummy-Instanz für Konstanten
        self.LOAD_R0_X = ds.LOAD_R0_X; self.LOAD_R1_X = ds.LOAD_R1_X; self.LOAD_R2_X = ds.LOAD_R2_X
        self.LOAD_AR_X = ds.LOAD_AR_X; self.MOVE_R0_R1 = ds.MOVE_R0_R1
        self.ALU_XOR_R1R2_R0 = ds.ALU_XOR_R1R2_R0; self.ALU_AND_R1R2_R0 = ds.ALU_AND_R1R2_R0
        self.ALU_OR_R1R2_R0 = ds.ALU_OR_R1R2_R0; self.ALU_NOT_R1_R0 = ds.ALU_NOT_R1_R0
        self.STORE_R0_MEM_AR = ds.STORE_R0_MEM_AR; self.LOAD_R0_MEM_AR = ds.LOAD_R0_MEM_AR
        self.INC_AR = ds.INC_AR; self.DEC_AR = ds.DEC_AR; self.JUMP_IF_ZF_ADDR = ds.JUMP_IF_ZF_ADDR
        self.CALL_ADDR = ds.CALL_ADDR; self.RETURN = ds.RETURN; self.OUT_R0 = ds.OUT_R0
        self.NO_OP = ds.NO_OP
        self.instr_names = ds.instr_names

        self.reset_state()

    def reset_state(self):
        self.r = [0.0] * 4  # R0, R1, R2, R3
        self.ar = 0.0
        self.zf, self.eqf = 0.0, 0.0
        self.data_memory = [0.0] * self.cpu_params["data_mem_size"]
        self.ret_stack = [0] * self.cpu_params["ret_stack_depth"]
        self.ret_sp = 0
        self.previous_y = 0.0
        self.program_counter = 0
        self.history = []

    def _decode_val_from_control_bits(self, bits_list):
        val = 0
        for i, bit in enumerate(bits_list): val += int(round(bit)) * (2**i)
        return val

    def _to_one_hot(self, value, num_classes): # Aus Dataset kopiert
        one_hot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= value < num_classes: one_hot[int(value)] = 1.0
        return one_hot


    def step(self, instr_idx_clean, control_data_bits_clean):
        # Aktuellen Zustand für History speichern (vor Ausführung)
        current_state_snapshot = {
            'pc': self.program_counter, 'instr': self.instr_names[instr_idx_clean], 'instr_idx': instr_idx_clean,
            'ctrl_bits': list(map(int,control_data_bits_clean)),
            'r': [int(val) for val in self.r], 'ar': int(self.ar), 'zf': int(self.zf), 'eqf': int(self.eqf),
            'mem': [int(val) for val in self.data_memory], 'sp': self.ret_sp,
            'stack_top': int(self.ret_stack[self.ret_sp-1]) if self.ret_sp > 0 else -1,
            'y_prev': int(self.previous_y)
        }

        # Logik aus dem Dataset __iter__ hier anwenden
        next_r = list(self.r); next_ar = self.ar; next_zf, next_eqf = self.zf, self.eqf
        next_data_memory = list(self.data_memory); target_y = 0.0; pc_jump_target = -1
        addr_from_bits = self._decode_val_from_control_bits(control_data_bits_clean) % (self.cpu_params["max_pc_val"] +1)

        if instr_idx_clean == self.LOAD_R0_X: next_r[0] = control_data_bits_clean[0]
        elif instr_idx_clean == self.LOAD_R1_X: next_r[1] = control_data_bits_clean[0]
        elif instr_idx_clean == self.LOAD_R2_X: next_r[2] = control_data_bits_clean[0]
        elif instr_idx_clean == self.LOAD_AR_X: next_ar = float(self._decode_val_from_control_bits(control_data_bits_clean[0:3]) % self.cpu_params["data_mem_size"])
        elif instr_idx_clean == self.MOVE_R0_R1: next_r[0] = self.r[1]
        elif instr_idx_clean == self.ALU_XOR_R1R2_R0: res = float(int(self.r[1]) ^ int(self.r[2])); next_r[0] = res; next_zf = 1.0 if res == 0 else 0.0; next_eqf = 1.0 if int(self.r[1])==int(self.r[2]) else 0.0
        elif instr_idx_clean == self.ALU_AND_R1R2_R0: res = float(int(self.r[1]) & int(self.r[2])); next_r[0] = res; next_zf = 1.0 if res == 0 else 0.0; next_eqf = 1.0 if int(self.r[1])==int(self.r[2]) else 0.0
        elif instr_idx_clean == self.ALU_OR_R1R2_R0: res = float(int(self.r[1]) | int(self.r[2])); next_r[0] = res; next_zf = 1.0 if res == 0 else 0.0; next_eqf = 1.0 if int(self.r[1])==int(self.r[2]) else 0.0
        elif instr_idx_clean == self.ALU_NOT_R1_R0: res = float(1 - int(self.r[1])); next_r[0] = res; next_zf = 1.0 if res == 0 else 0.0
        elif instr_idx_clean == self.STORE_R0_MEM_AR: next_data_memory[int(self.ar)] = self.r[0]
        elif instr_idx_clean == self.LOAD_R0_MEM_AR: next_r[0] = self.data_memory[int(self.ar)]
        elif instr_idx_clean == self.INC_AR: next_ar = (self.ar + 1) % self.cpu_params["data_mem_size"]
        elif instr_idx_clean == self.DEC_AR: next_ar = (self.ar - 1 + self.cpu_params["data_mem_size"]) % self.cpu_params["data_mem_size"]
        elif instr_idx_clean == self.JUMP_IF_ZF_ADDR:
            if int(self.zf) == 1: pc_jump_target = addr_from_bits
        elif instr_idx_clean == self.CALL_ADDR:
            if self.ret_sp < self.cpu_params["ret_stack_depth"]: self.ret_stack[self.ret_sp] = self.program_counter + 1; self.ret_sp = (self.ret_sp + 1)
            pc_jump_target = addr_from_bits
        elif instr_idx_clean == self.RETURN:
            if self.ret_sp > 0: self.ret_sp = self.ret_sp - 1; pc_jump_target = self.ret_stack[self.ret_sp]
        elif instr_idx_clean == self.OUT_R0: target_y = next_r[0]
        elif instr_idx_clean == self.NO_OP: target_y = self.previous_y

        # Zustände aktualisieren
        self.r, self.ar, self.zf, self.eqf = next_r, next_ar, next_zf, next_eqf
        self.data_memory = next_data_memory; self.previous_y = target_y
        if pc_jump_target != -1: self.program_counter = pc_jump_target
        else: self.program_counter = (self.program_counter + 1) % (self.cpu_params["max_pc_val"] +1)

        current_state_snapshot['y_out_expected'] = int(target_y)
        self.history.append(current_state_snapshot)
        return target_y

    def get_model_input_features(self, instr_idx_clean, control_data_bits_clean):
        """ Erstellt den Input-Vektor für das Modell basierend auf dem aktuellen CPU-Zustand. """
        instruction_one_hot = self._to_one_hot(instr_idx_clean, self.cpu_params["num_instructions"])
        ar_one_hot = self._to_one_hot(self.ar, self.cpu_params["data_mem_size"])
        # SP für One-Hot ist der aktuelle SP modulo Tiefe.
        # Wenn SP = ret_stack_depth, bedeutet das, der Stack ist "konzeptionell voll" und der nächste Push würde wrappen/überschreiben.
        # Für One-Hot-Encoding verwenden wir SP % Tiefe, damit der Index immer im Bereich liegt.
        current_sp_for_one_hot = self.ret_sp % self.cpu_params["ret_stack_depth"]
        sp_one_hot = self._to_one_hot(current_sp_for_one_hot, self.cpu_params["ret_stack_depth"])


        input_features_clean = control_data_bits_clean + \
                               list(instruction_one_hot) + \
                               self.r + \
                               list(ar_one_hot) + \
                               [self.zf, self.eqf] + \
                               self.data_memory + \
                               list(sp_one_hot) + \
                               [self.previous_y]
        return torch.tensor([input_features_clean], dtype=torch.float32).to(DEVICE)


def execute_program_on_model_and_sim(model, program, simulator, cpu_params_dict):
    """Führt ein festes Programm auf dem Modell und dem Simulator aus."""
    model.eval()
    simulator.reset_state()
    
    print("\n--- Programmausführung Start ---")
    print(f"{'PC':>3} | {'Instr':<7} | {'Ctrl':<6} | {'R0-R3':<10} | {'AR':>2} | {'ZF':>2} | {'EQF':>3} | {'Mem':<17} | {'SP':>2} | {'StkTop':>6} | {'y_prev':>6} | {'Exp_Y':>5} | {'Mod_Y':>5} | {'Alpha':>5}")
    print("-" * 120)

    correct_y_predictions = 0
    total_steps = len(program)

    for pc_target, (instr_idx, control_bits) in enumerate(program):
        # Sicherstellen, dass der Simulator PC mit dem erwarteten PC übereinstimmt (für Sprünge etc.)
        # Dies ist eine Vereinfachung; ein echter Test müsste den PC des Simulators verwenden.
        # Für dieses Beispiel nehmen wir an, dass das Programm sequentiell ist oder Sprünge
        # bereits in der `program`-Liste berücksichtigt sind (was nicht trivial ist).
        # Für eine robuste Auswertung müsste der Simulator den PC steuern.
        # Hier simulieren wir den PC extern für die Programmliste.

        # Hole aktuellen Zustand vom Simulator, um den Modell-Input zu bilden
        model_input_tensor = simulator.get_model_input_features(instr_idx, control_bits)

        # Modell-Vorhersage
        predicted_y = model.predict_next_state_and_output(model_input_tensor)
        mean_alpha = np.mean(model.last_alphas.cpu().numpy()) if model.last_alphas is not None else -1

        # Simulator-Schritt (aktualisiert internen Zustand des Simulators UND gibt erwartetes y zurück)
        expected_y = simulator.step(instr_idx, control_bits) # Simulator PC wird intern aktualisiert

        # History-Eintrag des Simulators holen (der *vor* dem aktuellen Step gemacht wurde)
        state_before_exec = simulator.history[-1]

        if int(predicted_y) == int(expected_y):
            correct_y_predictions += 1

        print(f"{state_before_exec['pc']:>3} | {state_before_exec['instr']:<7} | {''.join(map(str,state_before_exec['ctrl_bits'])) :<6} | {str(state_before_exec['r']):<10} | {state_before_exec['ar']:>2} | {state_before_exec['zf']:>2} | {state_before_exec['eqf']:>3} | {str(state_before_exec['mem']):<17} | {state_before_exec['sp']:>2} | {state_before_exec['stack_top']:>6} | {state_before_exec['y_prev']:>6} | {int(expected_y):>5} | {int(predicted_y):>5} | {mean_alpha:.3f}")

    accuracy = (correct_y_predictions / total_steps) * 100
    print("-" * 120)
    print(f"Programmausführung beendet. Genauigkeit der y_t Vorhersagen: {accuracy:.2f}% ({correct_y_predictions}/{total_steps})")
    return accuracy


# --- Hauptteil ---
if __name__ == "__main__":
    set_seed(SEED)

    num_instructions_task = 18
    data_mem_size_task = 8
    ret_stack_depth_task = 4
    pc_bits_task = 6
    input_size = 6 + num_instructions_task + 4 + data_mem_size_task + 2 + data_mem_size_task + ret_stack_depth_task + 1 # 51

    output_size = 1
    learning_rate = 0.0005
    batch_size = 4096 # War 4096 in Test20
    epochs = 15 # Reduziert für schnelleren Durchlauf dieses Beispiels, Original war 10, dann 500 geplant
    noise_level_data = 0.001
    target_accuracy_threshold = 0.90 # Angepasst für dieses Beispiel

    dpp_units = 128
    shared_g_dim_config = 25

    print(f"Task: CPU-Kern V0.1 - Programmausführungstest")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    num_train_sequences_per_epoch = 50000 # Reduziert für dieses Beispiel
    num_test_sequences_per_epoch = 10000
    seq_len_task = 50 # Kürzere Sequenzen für Training

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
    print(f"\nModellparameter: {param_count}")

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-8)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.2, min_lr=1e-9, verbose=True) # Angepasste Patience

    print(f"\n--- Training DPP SharedG für CPU-Kern V0.1 ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_CPU_Core_V0_1", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch, scheduler_obj=scheduler
    )

    print("\n--- Finale Evaluation des trainierten Modells (auf Zufallsdaten) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_CPU_Core_V0_1", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_CPU_Core_V0_1 - Parameter: {param_count}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")

    # --- Spezifisches Programm definieren und ausführen ---
    # Instruktions-Indizes für das Programm (aus CPU_Core_V0_1_Dataset)
    LDR0X = 0; LDR1X = 1; LDARX = 3; STOMEM = 9; LDMEM = 10; OUTR0 = 16; XOR = 5; AND = 6; NOOP = 17

    # Programm: Lade 1 in R0, Lade 0 in R1. Speichere R0 an Speicheradresse 0 (AR=0).
    # Lade R1 an Speicheradresse 1 (AR=1). Lade Wert von Mem[0] in R0. XOR R1 mit R0, Ergebnis in R0. Gib R0 aus.
    # Kontrollbits: [val_bit, c1,c2,c3,c4,c5] (für LOAD_RX_X ist val_bit relevant, für LOAD_AR_X c0-c2 relevant)
    # Für dieses Testprogramm vereinfachen wir:
    # - control_bits[0] ist der Wert für LOAD_RX_X
    # - control_bits[0:3] sind die Bits für LOAD_AR_X (Wert 0-7)
    # - control_bits sind für andere Operationen nicht direkt relevant für die Kernlogik, können aber als Kontext dienen.
    
    # Einfaches Programm: R0=1, R1=0, AR=0, Mem[0]=R0, AR=1, Mem[1]=R1, AR=0, R0=Mem[0], R1=Mem[1] (über Umweg, hier direkt für Test), R0 = R0 XOR R1, OUT R0
    # PC | INSTR        | CTRL_BITS   | Kommentar
    # 0  | LDR0X        | [1,0,0,0,0,0] | R0 = 1
    # 1  | LDR1X        | [0,0,0,0,0,0] | R1 = 0
    # 2  | LDARX        | [0,0,0,0,0,0] | AR = 0 (ctrl_bits[0:3] = 000)
    # 3  | STOMEM       | [0,0,0,0,0,0] | Mem[AR] (Mem[0]) = R0 (1)
    # 4  | LDARX        | [1,0,0,0,0,0] | AR = 1 (ctrl_bits[0:3] = 100)
    # 5  | STOMEM       | [0,0,0,0,0,0] | Mem[AR] (Mem[1]) = R1 (0)
    # 6  | LDARX        | [0,0,0,0,0,0] | AR = 0
    # 7  | LDMEM        | [0,0,0,0,0,0] | R0 = Mem[AR] (Mem[0]) -> R0=1
    # 8  | LDARX        | [1,0,0,0,0,0] | AR = 1
    # 9  | LDR1X        | [0,0,0,0,0,0] | Lade R1 explizit mit Mem[1] (hier simulieren wir das Laden von Mem[1] in R1 durch eine LOAD Anweisung)
    #    | LDMEM        | [0,0,0,0,0,0] | R1 = Mem[AR] (Mem[1]) -> R1=0 (Diese Instruktion würde R0 laden, nicht R1. Wir brauchen MOVE R0,R1 oder laden direkt.)
    #    Nehmen wir an, R1 ist schon 0.
    # 10 | XOR          | [0,0,0,0,0,0] | R0 = R0 (1) XOR R1 (0) -> R0 = 1
    # 11 | OUTR0        | [0,0,0,0,0,0] | Output R0 (1)
    # 12 | NOOP         | [0,0,0,0,0,0] |
    
    # Vereinfachtes Programm, um die Logik zu testen:
    test_program_sequence = [
    # (INSTRUKTION, [val_bit/addr0, addr1, addr2, c3, c4, c5])
        (LDR0X,       [1,0,0,0,0,0]), # R0 = 1
        (LDR1X,       [0,0,0,0,0,0]), # R1 = 0
        (XOR,         [0,0,0,0,0,0]), # R0 = R0 XOR R1 (1^0=1). ZF=0, EQF=0
        (OUTR0,       [0,0,0,0,0,0]), # Output R0 (sollte 1 sein)
        (LDR0X,       [1,0,0,0,0,0]), # R0 = 1
        (LDR1X,       [1,0,0,0,0,0]), # R1 = 1
        (XOR,         [0,0,0,0,0,0]), # R0 = R0 XOR R1 (1^1=0). ZF=1, EQF=1
        (OUTR0,       [0,0,0,0,0,0]), # Output R0 (sollte 0 sein)
        (LDARX,       [0,0,0,0,0,0]), # AR = 0
        (STOMEM,      [0,0,0,0,0,0]), # Mem[0] = R0 (0)
        (LDMEM,       [0,0,0,0,0,0]), # R0 = Mem[0] (0)
        (OUTR0,       [0,0,0,0,0,0]), # Output R0 (sollte 0 sein)
    ]


    cpu_sim = CPUSimulator(num_instructions=num_instructions_task,
                           data_mem_size=data_mem_size_task,
                           ret_stack_depth=ret_stack_depth_task,
                           pc_bits=pc_bits_task)

    execute_program_on_model_and_sim(model_to_train, test_program_sequence, cpu_sim, cpu_sim.cpu_params)