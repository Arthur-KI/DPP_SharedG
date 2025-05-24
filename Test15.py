import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 55 # Neuer Seed für diesen ultimativen Test
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
        # Optional: Aktivierung auf x_shared_g
        # x_shared_g = torch.relu(x_shared_g)
        g_logits=F.linear(x_shared_g,self.w_g_unit,self.b_g_unit)
        alpha=torch.sigmoid(g_logits); z_final=alpha*z_a+(1-alpha)*z_b
        if return_alpha: return z_final,alpha
        return z_final

# --- Modell-Definitionen (DPPModelBase bleibt gleich) ---
class DPPModelBase(nn.Module):
    def __init__(self, dpp_layer, hidden_size_dpp, output_size):
        super(DPPModelBase, self).__init__(); self.dpp_layer1 = dpp_layer
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
        out = self.fc_out(out)
        return out

# --- Datengenerierung für "Stack-Maschine V1" ---
class StackMachineV1Dataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, 
                 num_instructions=13, stack_levels=4):
        super(StackMachineV1Dataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.num_instructions = num_instructions
        self.stack_levels = stack_levels # Größe des Stacks (Anzahl Ebenen)

        # Instruktions-Indizes
        self.LOAD_R0_X = 0
        self.LOAD_R1_X = 1
        self.MOVE_R0_R1 = 2
        self.XOR_R0_R1_R0 = 3
        self.ADD_R0_R1_R0 = 4
        self.NOT_R0 = 5
        self.PUSH_R0 = 6
        self.POP_R0 = 7
        self.SET_FLAG_IF_R0_ZERO = 8
        self.JUMP_IF_F0_SKIP_2 = 9
        self.OUT_R0 = 10
        self.OUT_STACK_TOP = 11
        self.NO_OP = 12

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            r0, r1, f0 = 0.0, 0.0, 0.0
            sp = 0 # Stack Pointer, zeigt auf das nächste freie Element (0 bis stack_levels-1)
            stack = [0.0] * self.stack_levels
            previous_y = 0.0
            
            program_counter = 0
            skip_counter = 0

            while program_counter < self.seq_len:
                current_instr_idx_clean = 0 # Wird unten gesetzt
                current_x1_bit_clean = 0.0

                if skip_counter > 0:
                    current_instr_idx_clean = self.NO_OP
                    current_x1_bit_clean = 0.0 # Irrelevant für NO_OP
                    skip_counter -=1
                else:
                    current_x1_bit_clean = float(np.random.randint(0, 2))
                    current_instr_idx_clean = np.random.randint(0, self.num_instructions)
                
                instruction_one_hot = self._to_one_hot(current_instr_idx_clean, self.num_instructions)
                sp_one_hot = self._to_one_hot(sp, self.stack_levels)

                # Inputs für das Modell (mit Rauschen)
                x1_t_noisy = current_x1_bit_clean + np.random.normal(0, self.noise_level)
                r0_tm1_noisy = r0 + np.random.normal(0, self.noise_level)
                r1_tm1_noisy = r1 + np.random.normal(0, self.noise_level)
                f0_tm1_noisy = f0 + np.random.normal(0, self.noise_level)
                sp_one_hot_noisy = sp_one_hot # Kein Rauschen auf One-Hot SP für Einfachheit
                stack_noisy = [s_val + np.random.normal(0, self.noise_level) for s_val in stack]
                y_tm1_noisy = previous_y + np.random.normal(0, self.noise_level)

                input_features = ([x1_t_noisy] + list(instruction_one_hot) + 
                                  [r0_tm1_noisy, r1_tm1_noisy, f0_tm1_noisy] + 
                                  list(sp_one_hot_noisy) + stack_noisy + [y_tm1_noisy])
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Nächste Zustände und Ziel-y (basierend auf sauberen Werten von t-1)
                next_r0, next_r1, next_f0 = r0, r1, f0
                next_sp = sp
                next_stack = list(stack)
                target_y = 0.0 

                if current_instr_idx_clean == self.LOAD_R0_X: next_r0 = current_x1_bit_clean
                elif current_instr_idx_clean == self.LOAD_R1_X: next_r1 = current_x1_bit_clean
                elif current_instr_idx_clean == self.MOVE_R0_R1: next_r0 = r1
                elif current_instr_idx_clean == self.XOR_R0_R1_R0: next_r0 = float(int(r0) ^ int(r1))
                elif current_instr_idx_clean == self.ADD_R0_R1_R0: next_r0 = float(int(r0) ^ int(r1))
                elif current_instr_idx_clean == self.NOT_R0: next_r0 = float(1 - int(r0))
                elif current_instr_idx_clean == self.PUSH_R0:
                    if sp < self.stack_levels: # Nur pushen, wenn Platz ist (Überlauf nicht behandelt)
                        next_stack[sp] = r0
                        next_sp = (sp + 1) % self.stack_levels # Einfaches Wrap-Around oder Fehler bei Überlauf
                    # Um sicherzustellen, dass SP immer gültig ist, auch wenn Stack voll ist
                    if sp == self.stack_levels -1 and next_sp == 0 : # Wenn SP am Ende war und umbricht
                         pass # SP bleibt am Ende, wenn Stack voll wäre, oder wir implementieren Fehler
                    elif sp < self.stack_levels -1 :
                         next_sp = sp + 1
                    # else: SP bleibt, Stack ist voll. Für Einfachheit hier: SP wraps around.
                    next_sp = (sp + 1) % self.stack_levels


                elif current_instr_idx_clean == self.POP_R0:
                    if sp > 0: # Nur poppen, wenn etwas auf dem Stack ist
                        next_sp = (sp - 1 + self.stack_levels) % self.stack_levels
                        next_r0 = stack[next_sp] 
                    # Um sicherzustellen, dass SP immer gültig ist
                    elif sp == 0 and next_sp == self.stack_levels -1 : # Wenn SP am Anfang war und umbricht
                        next_r0 = stack[next_sp] # Holt sich das "letzte" Element
                    # else: SP bleibt 0, Stack ist leer. R0 wird nicht geändert.
                    if sp > 0 :
                        next_sp = sp -1
                    elif sp == 0: # Stack war leer, versuche vom "Ende" zu poppen (wrap around)
                        next_sp = self.stack_levels -1
                        next_r0 = stack[next_sp]

                elif current_instr_idx_clean == self.SET_FLAG_IF_R0_ZERO:
                    next_f0 = 1.0 if int(r0) == 0 else 0.0
                elif current_instr_idx_clean == self.JUMP_IF_F0_SKIP_2:
                    if int(f0) == 1: skip_counter = 2 
                elif current_instr_idx_clean == self.OUT_R0: target_y = next_r0 
                elif current_instr_idx_clean == self.OUT_STACK_TOP:
                    if sp > 0: target_y = stack[(sp - 1 + self.stack_levels) % self.stack_levels]
                    elif sp == 0 and len(stack) > 0 : target_y = stack[self.stack_levels -1 ] # Wrap around
                    else: target_y = 0.0 # Leerer Stack
                elif current_instr_idx_clean == self.NO_OP: target_y = previous_y

                target_tensor = torch.tensor([target_y], dtype=torch.float32)
                yield input_vector, target_tensor

                r0, r1, f0 = next_r0, next_r1, next_f0
                sp = next_sp
                stack = next_stack
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

        if (accuracy>=target_accuracy and epoch_at_target_accuracy is None) or (epoch+1)%(epochs//20 if epochs>=20 else 1)==0 or epoch==0 or epoch==epochs-1 : # Print every 5% or so
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
    stack_levels_task = 4
    # Input: x1(1) + instr(13) + R0(1) + R1(1) + F0(1) + SP_oh(4) + Stack(4) + y_tm1(1) = 26
    input_size = 1 + num_instructions_task + 2 + 1 + stack_levels_task + stack_levels_task + 1

    output_size = 1
    learning_rate = 0.001 # Evtl. anpassen
    batch_size = 2048 # Große Batch Size
    epochs = 5      # sehr wenig Epochen
    noise_level_data = 0.001 # Fast saubere Daten
    target_accuracy_threshold = 0.65 # Sehr konservatives Ziel

    # Modellkonfigurationen
    dpp_units = 64
    shared_g_dim_config = 13 # input_size (26) / 2

    print(f"Task: Stack-Maschine V1 (Registers, Stack, Flag, Conditional Jump)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 50000 # Sehr viele Daten
    num_test_sequences_per_epoch = 10000
    seq_len_task = 60 # Lange Programme

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")

    train_dataset = StackMachineV1Dataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, stack_levels_task)
    test_dataset = StackMachineV1Dataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, num_instructions_task, stack_levels_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    param_count = count_parameters(model_to_train)
    print(f"\nModellparameter: {param_count}") # Sollte ~4768 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-7) # Sehr kleiner WD
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=40, factor=0.2, min_lr=1e-7, verbose=True) 

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
    max_inspect_alpha_stack_v1 = 30 

    inspector_dataset_stack_v1 = StackMachineV1Dataset(1, max_inspect_alpha_stack_v1 + seq_len_task, 0.0, num_instructions_task, stack_levels_task)
    inspector_loader_stack_v1 = DataLoader(inspector_dataset_stack_v1, batch_size=1, shuffle=False)
    sample_count_inspect = 0

    instr_names_stack_v1 = [
        "LOAD_R0_X", "LOAD_R1_X", "MOVE_R0_R1", "XOR_R0_R1_R0", "ADD_R0_R1_R0",
        "NOT_R0", "PUSH_R0", "POP_R0", "SET_F0_R0_0", "JMP_IF_F0_SK2",
        "OUT_R0", "OUT_STK_TOP", "NO_OP"
    ]
    print(f"Format Input: [x1, instr_oh({num_instructions_task}), R0,R1,F0, SP_oh({stack_levels_task}), Stack({stack_levels_task}), y-1]")
    print("----------------------------------------------------------------------------------------------------")

    with torch.no_grad():
        # Zustandsvariablen für die Inspektion
        insp_r0, insp_r1, insp_f0 = 0.0, 0.0, 0.0
        insp_sp = 0
        insp_stack = [0.0] * stack_levels_task
        insp_previous_y = 0.0
        insp_skip_counter = 0

        for insp_idx, (insp_input_vec_model_ignored, insp_target_vec_model_ignored) in enumerate(inspector_loader_stack_v1): # Dataset generiert die Sequenz
            if sample_count_inspect >= max_inspect_alpha_stack_v1: break
            
            current_insp_x1_clean = float(random.randint(0,1))
            current_instr_idx_clean = 0
            if insp_skip_counter > 0:
                current_instr_idx_clean = 12 # NO_OP
                insp_skip_counter -= 1
            else:
                current_instr_idx_clean = random.randint(0, num_instructions_task -1)

            current_instr_oh_clean = np.zeros(num_instructions_task, dtype=np.float32)
            current_instr_oh_clean[current_instr_idx_clean] = 1.0
            current_sp_oh_clean = np.zeros(stack_levels_task, dtype=np.float32)
            current_sp_oh_clean[insp_sp] = 1.0

            # Das sind die sauberen Inputs für das Modell für diesen Inspektionsschritt
            model_input_features_clean = ([current_insp_x1_clean] + list(current_instr_oh_clean) +
                                     [insp_r0, insp_r1, insp_f0] +
                                     list(current_sp_oh_clean) + list(insp_stack) + [insp_previous_y])
            inp_tensor_clean = torch.tensor([model_input_features_clean], dtype=torch.float32).to(DEVICE)

            # Modell-Prädiktion und Alphas
            _ = model_to_train(inp_tensor_clean, return_alpha_flag=True) # Für Alphas
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue
            
            model_output_logit = model_to_train(inp_tensor_clean) # Für Output
            model_output_y = float(torch.sigmoid(model_output_logit[0]).item() > 0.5)

            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            # Berechne den korrekten nächsten Zustand und Ziel-y (Logik aus Dataset)
            next_insp_r0, next_insp_r1, next_insp_f0 = insp_r0, insp_r1, insp_f0
            next_insp_sp = insp_sp
            next_insp_stack = list(insp_stack)
            correct_target_y = 0.0

            if current_instr_idx_clean == 0: next_insp_r0 = current_insp_x1_clean
            elif current_instr_idx_clean == 1: next_insp_r1 = current_insp_x1_clean
            elif current_instr_idx_clean == 2: next_insp_r0 = insp_r1
            elif current_instr_idx_clean == 3: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1))
            elif current_instr_idx_clean == 4: next_insp_r0 = float(int(insp_r0) ^ int(insp_r1))
            elif current_instr_idx_clean == 5: next_insp_r0 = float(1 - int(insp_r0))
            elif current_instr_idx_clean == 6: # PUSH_R0
                if insp_sp < stack_levels_task: next_insp_stack[insp_sp] = insp_r0
                next_insp_sp = (insp_sp + 1) % stack_levels_task
            elif current_instr_idx_clean == 7: # POP_R0
                if insp_sp > 0 : next_insp_sp = insp_sp - 1
                elif insp_sp == 0 : next_insp_sp = stack_levels_task -1 # Wrap around
                next_insp_r0 = insp_stack[next_insp_sp]
            elif current_instr_idx_clean == 8: next_insp_f0 = 1.0 if int(insp_r0) == 0 else 0.0
            elif current_instr_idx_clean == 9:
                if int(insp_f0) == 1: insp_skip_counter = 2 
            elif current_instr_idx_clean == 10: correct_target_y = next_insp_r0 
            elif current_instr_idx_clean == 11: # OUT_STACK_TOP
                current_top_sp = (insp_sp - 1 + stack_levels_task) % stack_levels_task
                if insp_sp > 0 : correct_target_y = insp_stack[current_top_sp] # If sp was >0, new sp is sp-1, element is at new_sp
                elif insp_sp == 0 and len(insp_stack) > 0 : correct_target_y = insp_stack[stack_levels_task-1]
                else: correct_target_y = 0.0
            elif current_instr_idx_clean == 12: correct_target_y = insp_previous_y
            
            stack_str = ",".join(map(str, [int(s) for s in insp_stack]))
            print(f"S{sample_count_inspect+1}: x1:{int(current_insp_x1_clean)},I:{instr_names_stack_v1[current_instr_idx_clean]}({current_instr_idx_clean})|R0i:{int(insp_r0)},R1i:{int(insp_r1)},F0i:{int(insp_f0)},SPi:{insp_sp},Stk:[{stack_str}],y-1i:{int(insp_previous_y)}|Pred_y:{int(model_output_y)},Tgt_y:{int(correct_target_y)},MA:{mean_alpha:.3f}")
            sample_count_inspect += 1

            # Update Zustände für den nächsten Inspektionsschritt
            insp_r0, insp_r1, insp_f0 = next_insp_r0, next_insp_r1, next_insp_f0
            insp_sp = next_insp_sp
            insp_stack = next_insp_stack
            insp_previous_y = correct_target_y

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")