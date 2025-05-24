import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 50 # Neuer Seed für diesen sehr komplexen Test
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

# --- Datengenerierung für "Multibit-Operation mit Zustands- und Kontexthistorie" ---
class ComplexStatefulOperationDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, counter_modulo=3):
        super(ComplexStatefulOperationDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.counter_modulo = counter_modulo

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(value)] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            current_x1_bits = np.random.randint(0, 2, self.seq_len)
            current_x2_bits = np.random.randint(0, 2, self.seq_len)
            current_c_bits = np.random.randint(0, 2, self.seq_len) # Kontrollbit für Reset

            previous_y_t_minus_1 = 0.0 # Initialer Output y_{-1}
            previous_y_t_minus_2 = 0.0 # Initialer Output y_{-2}
            previous_z = 0   # Initialer Zählerstand z_{-1}

            for t in range(self.seq_len):
                x1_t = float(current_x1_bits[t])
                x2_t = float(current_x2_bits[t])
                c_t = float(current_c_bits[t])

                # Rauschen zu den aktuellen Inputs hinzufügen
                x1_t_noisy = x1_t + np.random.normal(0, self.noise_level)
                x2_t_noisy = x2_t + np.random.normal(0, self.noise_level)
                c_t_noisy = c_t + np.random.normal(0, self.noise_level)
                # y_{t-1} und y_{t-2} als sauber für diesen Test annehmen (oder eigenes Rauschen hinzufügen)
                y_t_minus_1_noisy = previous_y_t_minus_1
                y_t_minus_2_noisy = previous_y_t_minus_2

                zt_minus_1_one_hot = self._to_one_hot(previous_z, self.counter_modulo)

                input_features = [x1_t_noisy, x2_t_noisy, c_t_noisy, y_t_minus_1_noisy, y_t_minus_2_noisy] + list(zt_minus_1_one_hot)
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Zielberechnung y_t basierend auf sauberen x1_t, x2_t, y_{t-1}, y_{t-2} und z_{t-1} (previous_z)
                if previous_z == 0:
                    target_y = float(int(x1_t) ^ int(x2_t)) # x1 XOR x2
                elif previous_z == 1:
                    target_y = float( (int(x1_t) & int(x2_t)) ^ int(previous_y_t_minus_1) ) # (x1 AND x2) XOR y_{t-1}
                else: # previous_z == 2
                    target_y = float( (int(x1_t) | int(x2_t)) ^ int(previous_y_t_minus_2) ) # (x1 OR x2) XOR y_{t-2}
                target_tensor = torch.tensor([target_y], dtype=torch.float32)

                yield input_vector, target_tensor

                # Update Zustände für den nächsten Schritt
                previous_y_t_minus_2 = previous_y_t_minus_1
                previous_y_t_minus_1 = target_y

                if int(c_t) == 1: # Reset
                    previous_z = 0
                elif int(x1_t) == 1 : # Inkrement nur wenn x1_t=1 und kein Reset
                    previous_z = (previous_z + 1) % self.counter_modulo
                # ansonsten bleibt previous_z gleich

    def __len__(self):
        return self.num_sequences_per_epoch * self.seq_len

# --- Trainings- und Evaluierungsfunktionen mit AMP (unverändert) ---
def train_model_amp(model, train_loader, criterion, optimizer, scaler, epochs=100, model_name="Model", target_accuracy=0.98, steps_per_epoch=None):
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
        time_this_epoch=time.time()-epoch_start_time # KORREKTUR: war total_training_start_time
        avg_epoch_loss=epoch_loss/num_batches if num_batches > 0 else 0
        accuracy=correct_preds/total_preds if total_preds > 0 else 0
        history['loss'].append(avg_epoch_loss); history['accuracy'].append(accuracy); history['time_per_epoch'].append(time_this_epoch)
        if (accuracy>=target_accuracy and epoch_at_target_accuracy is None) or (epoch+1)%(epochs//10 if epochs>=10 else 1)==0 or epoch==0:
             current_total_time=time.time()-total_training_start_time
             print(f"Epoch [{epoch+1}/{epochs}] {model_name}, Loss: {avg_epoch_loss:.4f}, Acc: {accuracy:.4f}, Total Time: {current_total_time:.2f}s")
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

    input_size = 8  # x1, x2, c, y_t-1, y_t-2, z_t-1 (one-hot, 3 bits)
    output_size = 1 # y_t
    learning_rate = 0.003
    batch_size = 512 # Größere Batch Size für mehr Samples pro Schritt
    epochs = 150     # Mehr Epochen für die komplexere Aufgabe
    noise_level_data = 0.05
    target_accuracy_threshold = 0.95 # Ziel etwas gesenkt für diese Komplexität
    counter_modulo_task = 3

    # Modellkonfigurationen
    dpp_units = 16
    shared_g_dim_config = 4 # input_size=8. 8/2=4.

    print(f"Task: Multibit-Operation mit Zustands- und Kontexthistorie (Zähler Mod {counter_modulo_task})")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 4000 # Erhöhte Anzahl an Sequenzen
    num_test_sequences_per_epoch = 800
    seq_len_task = 30 # Längere Sequenzen

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size
    print(f"Train steps per epoch: {train_steps_per_epoch}, Test steps per epoch: {test_steps_per_epoch}")


    train_dataset = ComplexStatefulOperationDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, counter_modulo=counter_modulo_task)
    test_dataset = ComplexStatefulOperationDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, counter_modulo=counter_modulo_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    print(f"\nModellparameter: {count_parameters(model_to_train)}") # Sollte jetzt ~530 sein

    optimizer = optim.AdamW(model_to_train.parameters(), lr=learning_rate, weight_decay=1e-5) # AdamW mit leichtem WD
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    # Scheduler, um LR ggf. anzupassen
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.2, min_lr=1e-6, verbose=True)


    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für komplexere Aufgabe ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler, # Scheduler wird intern im Loop verwendet
        epochs=epochs, model_name="DPP_ComplexStatefulOp", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch
    )

    print("\n--- Finale Evaluation (Komplexere Aufgabe) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_ComplexStatefulOp", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_ComplexStatefulOp - Parameter: {count_parameters(model_to_train)}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Komplexere Aufgabe) ---")
    model_to_train.eval()
    test_iter = iter(test_loader) # Neu erstellen für frische Samples
    sample_count_inspect = 0
    max_inspect = 20
    print(f"Format Input: [x1_n, x2_n, c_n, y(t-1)_n, y(t-2)_n, z(t-1)_oh0, z(t-1)_oh1, z(t-1)_oh2]")
    print(f"Logik z_t: 0 if c_t=1, else ( (z(t-1)+1)%{counter_modulo_task} if x1_t=1 else z(t-1) )")
    print(f"Logik y_t: (x1 XOR x2) if z(t-1)=0, ((x1 AND x2) XOR y(t-1)) if z(t-1)=1, ((x1 OR x2) XOR y(t-2)) if z(t-1)=2")
    print("----------------------------------------------------------------------------------")

    # Für die Inspektion generieren wir einige wenige, kontrollierte Sequenzen ohne Rauschen,
    # um die Logik der Alphas besser nachvollziehen zu können.
    inspector_dataset = ComplexStatefulOperationDataset(5, seq_len_task, 0.0, counter_modulo_task) # 5 Sequenzen, kein Rauschen
    inspector_loader = DataLoader(inspector_dataset, batch_size=1, shuffle=False) # Batch size 1 für einzelne Samples

    current_z_display = 0 # Hilfsvariable für die Anzeige des Zählerstands vor der Operation
    current_y_minus_1_display = 0
    current_y_minus_2_display = 0

    with torch.no_grad():
        for insp_input_vec, insp_target_vec in inspector_loader:
            if sample_count_inspect >= max_inspect: break

            inp_tensor = insp_input_vec.to(DEVICE)
            _ = model_to_train(inp_tensor, return_alpha_flag=True)
            alphas_batch = model_to_train.last_alphas
            if alphas_batch is None: continue

            inp_numpy = insp_input_vec[0].cpu().numpy() # Batch size ist 1
            lbl_numpy = insp_target_vec[0].cpu().item()
            alpha_vals = alphas_batch[0].cpu().numpy()
            mean_alpha = np.mean(alpha_vals)

            x1_disp = int(round(inp_numpy[0]))
            x2_disp = int(round(inp_numpy[1]))
            c_disp = int(round(inp_numpy[2]))
            # Die y(t-1) und y(t-2) im Input sind die *sauberen* Werte aus der Datengenerierung
            y_tm1_disp = int(round(inp_numpy[3]))
            y_tm2_disp = int(round(inp_numpy[4]))
            zt_minus_1_one_hot_part = inp_numpy[5:]
            zt_minus_1_disp = np.argmax(zt_minus_1_one_hot_part)

            # Bestimme den erwarteten Modus für die Ausgabe
            op_mode = ""
            if zt_minus_1_disp == 0: op_mode = "XOR(x1,x2)"
            elif zt_minus_1_disp == 1: op_mode = "AND(x1,x2)XORy-1"
            elif zt_minus_1_disp == 2: op_mode = "OR(x1,x2)XORy-2"


            print(f"S{sample_count_inspect+1}: In(x1,x2,c|y-1,y-2|z-1)~[{x1_disp},{x2_disp},{c_disp}|{y_tm1_disp},{y_tm2_disp}|{zt_minus_1_disp}]->Op:{op_mode}, Target:{lbl_numpy:.0f}, MeanAlpha:{mean_alpha:.3f}") # Alphas:{np.round(alpha_vals,2)}")
            sample_count_inspect += 1

    if sample_count_inspect == 0: print("Keine Samples für Inspektion gefunden.")