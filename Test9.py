import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 49 # Neuer Seed für neuen komplexeren Test
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

# --- Datengenerierung für "Zustandsgesteuerter Zähler mit Reset und Operation" ---
class StatefulCounterOperationDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, counter_modulo=4):
        super(StatefulCounterOperationDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.counter_modulo = counter_modulo

    def _to_one_hot(self, value, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[value] = 1.0
        return one_hot

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            current_x_bits = np.random.randint(0, 2, self.seq_len)
            current_c_bits = np.random.randint(0, 2, self.seq_len) # Kontrollbit für Reset
            
            previous_y = 0.0 # Initialer Output y_{-1}
            previous_z = 0   # Initialer Zählerstand z_{-1}

            for t in range(self.seq_len):
                xt = float(current_x_bits[t])
                ct = float(current_c_bits[t])

                # Berechne aktuellen Zählerstand z_t basierend auf z_{t-1} und c_t
                # ABER: der Input für das Modell ist z_{t-1} (der Zählerstand *vor* der aktuellen Aktion)
                # Der Output y_t hängt von z_{t-1} ab.
                # Für den *nächsten* Zeitschritt wird z_t dann zu z_{t-1}.

                xt_noisy = xt + np.random.normal(0, self.noise_level)
                ct_noisy = ct + np.random.normal(0, self.noise_level)
                yt_minus_1_noisy = previous_y # Annahme: sauber für diesen Test
                
                # z_{t-1} als One-Hot-Encoding
                zt_minus_1_one_hot = self._to_one_hot(previous_z, self.counter_modulo)
                # Optional: Rauschen zu One-Hot (vorsichtig, kann die Bedeutung zerstören)
                # zt_minus_1_one_hot_noisy = zt_minus_1_one_hot + np.random.normal(0, self.noise_level/5, size=zt_minus_1_one_hot.shape)


                input_features = [xt_noisy, ct_noisy, yt_minus_1_noisy] + list(zt_minus_1_one_hot) # oder _noisy
                input_vector = torch.tensor(input_features, dtype=torch.float32)

                # Zielberechnung y_t basierend auf x_t, y_{t-1} und z_{t-1} (previous_z)
                if previous_z == 0 or previous_z == 1: # Zählerstand niedrig -> XOR-Logik
                    target_y = float(int(xt) ^ int(previous_y))
                else: # Zählerstand hoch (2 oder 3) -> AND-Logik
                    target_y = float(int(xt) & int(previous_y))
                target_tensor = torch.tensor([target_y], dtype=torch.float32)

                yield input_vector, target_tensor

                # Update Zustände für den nächsten Schritt
                previous_y = target_y
                if int(ct) == 1: # Reset
                    previous_z = 0
                else: # Inkrement
                    previous_z = (previous_z + 1) % self.counter_modulo
                    
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
        time_this_epoch=time.time()-total_training_start_time
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

    input_size = 7  # xt, ct, yt-1, zt-1 (one-hot, 4 bits)
    output_size = 1 # yt
    learning_rate = 0.003 # Ggf. etwas niedriger für komplexere Aufgabe
    batch_size = 256
    epochs = 100     # Mehr Epochen
    noise_level_data = 0.05
    target_accuracy_threshold = 0.98 # Etwas niedrigeres Ziel für komplexere Aufgabe
    counter_modulo_task = 4

    # Modellkonfigurationen
    dpp_units = 16 # Mehr Units für komplexere Aufgabe
    shared_g_dim_config = 4 # z.B. input_size / 2 (abgerundet) + 1

    print(f"Task: Zustandsgesteuerter Zähler (Mod {counter_modulo_task}) mit Reset und Operation")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 2000 # Mehr Daten
    num_test_sequences_per_epoch = 400
    seq_len_task = 25 # Längere Sequenzen

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size

    train_dataset = StatefulCounterOperationDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, counter_modulo=counter_modulo_task)
    test_dataset = StatefulCounterOperationDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, counter_modulo=counter_modulo_task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    print(f"\nModellparameter: {count_parameters(model_to_train)}")

    optimizer = optim.Adam(model_to_train.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * train_steps_per_epoch, eta_min=1e-7)


    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Zustandsgest. Zähler ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_CounterOp", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch
    )

    print("\n--- Finale Evaluation (Zustandsgest. Zähler) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_CounterOp", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_CounterOp - Parameter: {count_parameters(model_to_train)}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Zustandsgest. Zähler) ---")
    model_to_train.eval()
    test_iter = iter(test_loader)
    sample_count_inspect = 0
    max_inspect = 20
    print(f"Format Input: [xt_n, ct_n, yt-1_n, zt-1_oh0, zt-1_oh1, zt-1_oh2, zt-1_oh3]")
    print(f"Logik y_t: (xt XOR yt-1) if zt-1 in [0,1], else (xt AND yt-1)")
    print(f"Logik z_t: 0 if ct=1, else (zt-1 + 1)%{counter_modulo_task}")
    print("Alpha-Erwartung (Hypothese): Pfad A lernt XOR, Pfad B lernt AND.")
    print("  Wenn zt-1 in [0,1] (XOR-Modus), sollte Alpha -> 1.")
    print("  Wenn zt-1 in [2,3] (AND-Modus), sollte Alpha -> 0.")
    print("----------------------------------------------------------------------------------")
    with torch.no_grad():
        # Temporäre Datengenerierung nur für Inspektion, um saubere Werte zu haben
        clean_inspector_dataset = StatefulCounterOperationDataset(1, max_inspect + 5, 0.0, counter_modulo_task) # Kein Rauschen
        clean_iter = iter(clean_inspector_dataset)
        
        current_z_state_for_print = 0 # Nur für die Print-Logik

        for _ in range(max_inspect):
            try:
                # Hole saubere Inputs und das zugehörige saubere Label für die Anzeige
                # Wir nehmen die Inputs, die für das Modell generiert werden würden, aber ohne Rauschen
                # Dies erfordert, die Logik der Datengenerierung hier etwas nachzubilden für die Anzeige
                
                # Erzeuge einen Batch mit nur einem Sample für die Alpha-Inspektion
                # Die Logik in StatefulCounterOperationDataset ist ein Generator, daher etwas umständlich
                single_sample_loader = DataLoader(StatefulCounterOperationDataset(1,1,0.0, counter_modulo_task), batch_size=1)
                inp_tensor, lbl_tensor = next(iter(single_sample_loader))
                inp_tensor, lbl_tensor = inp_tensor.to(DEVICE), lbl_tensor.to(DEVICE)

                _ = model_to_train(inp_tensor, return_alpha_flag=True)
                alphas_batch = model_to_train.last_alphas
                if alphas_batch is None: continue

                inp_numpy = inp_tensor[0].cpu().numpy() # Nur das eine Sample im Batch
                lbl_numpy = lbl_tensor[0].cpu().item()
                alpha_vals = alphas_batch[0].cpu().numpy()
                mean_alpha = np.mean(alpha_vals)

                # Extrahiere die ursprünglichen, uncodierten Werte für die Anzeige
                xt_clean = round(inp_numpy[0]) # Annahme: Rauschen war klein genug
                ct_clean = round(inp_numpy[1])
                yt_minus_1_clean = round(inp_numpy[2])
                # Finde den one-hot encodierten z_{t-1} Wert
                zt_minus_1_one_hot_part = inp_numpy[3:]
                zt_minus_1_clean = np.argmax(zt_minus_1_one_hot_part) if np.sum(zt_minus_1_one_hot_part) > 0 else -1 # -1 falls kein Bit gesetzt

                print(f"Sample {sample_count_inspect+1}: In(x,c,y-1,z-1)~[{xt_clean},{ct_clean},{yt_minus_1_clean},{zt_minus_1_clean}], Target:{lbl_numpy:.0f}, MeanAlpha:{mean_alpha:.3f}, Alphas:{np.round(alpha_vals,2)}")
                sample_count_inspect += 1
            except StopIteration:
                break
    if sample_count_inspect == 0: print("Keine Samples im Testloader für Inspektion gefunden.")