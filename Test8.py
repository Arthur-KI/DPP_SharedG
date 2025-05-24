import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import time
import random

# --- Seed-Initialisierung ---
SEED = 48 # Neuer Seed für neuen Test
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

# --- Datengenerierung für "Zustandsgesteuerte Operation mit Gedächtnis" ---
class StatefulOperationDataset(IterableDataset):
    def __init__(self, num_sequences_per_epoch, seq_len, noise_level=0.0, fixed_bit_A=1):
        super(StatefulOperationDataset).__init__()
        self.num_sequences_per_epoch = num_sequences_per_epoch
        self.seq_len = seq_len
        self.noise_level = noise_level
        self.fixed_bit_A = float(fixed_bit_A) # Das Bit für den XOR-Modus

    def __iter__(self):
        for _ in range(self.num_sequences_per_epoch):
            current_x_bits = np.random.randint(0, 2, self.seq_len)
            previous_y = 0.0 # Initialer Akkumulatorwert/Zustand

            for t in range(self.seq_len):
                xt = float(current_x_bits[t])

                xt_noisy = xt + np.random.normal(0, self.noise_level)
                yt_minus_1_noisy = previous_y # Annahme: vorheriger Output ist sauber oder hat eigenes Rauschen
                                             # Für diesen Test nehmen wir ihn als sauber an.

                input_vector = torch.tensor([xt_noisy, yt_minus_1_noisy], dtype=torch.float32)

                # Zielberechnung basierend auf sauberen Bits
                if int(previous_y) == 0: # Zustand "XOR-Modus" mit fixed_bit_A
                    target_y = float(int(xt) ^ int(self.fixed_bit_A)) # x_t XOR fixed_bit_A
                else: # Zustand "Durchleit-Modus"
                    target_y = xt # y_t = x_t

                target_tensor = torch.tensor([target_y], dtype=torch.float32)

                yield input_vector, target_tensor
                previous_y = target_y # Update für den nächsten Schritt

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
        time_this_epoch=time.time()-epoch_start_time
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

    input_size = 2  # xt, yt-1
    output_size = 1 # yt
    learning_rate = 0.005
    batch_size = 256
    epochs = 60 # Ggf. mehr Epochen für komplexere Logik
    noise_level_data = 0.05
    target_accuracy_threshold = 0.99
    fixed_bit_A_for_XOR_mode = 1 # Das feste Bit für den XOR-Modus (kann 0 oder 1 sein)

    # Modellkonfigurationen
    dpp_units = 8
    shared_g_dim_config = 2 # (input_size / 2) wäre 1. Wir nehmen 2 für etwas mehr Kapazität im Gating.
    # DPP SharedG (input=2, dpp_units=8, shared_g_dim=2): (2*8+8)*2 + (2*2+2) + (2*8+8) = 48 + 6 + 24 = 78 params
    # + FC(8,1) = 9 params -> Total ~87

    print(f"Task: Zustandsgesteuerte Operation (y_t = (x_t XOR {fixed_bit_A_for_XOR_mode}) if y_t-1=0 else x_t)")
    print(f"Input size: {input_size}, Noise level (std dev): {noise_level_data}, AMP: {USE_AMP_GLOBAL}")
    print(f"DPP units: {dpp_units}, Shared Gating Dim: {shared_g_dim_config}")

    # Datengenerierung
    num_train_sequences_per_epoch = 1000
    num_test_sequences_per_epoch = 200
    seq_len_task = 20 # Längere Sequenzen

    train_steps_per_epoch = (num_train_sequences_per_epoch * seq_len_task) // batch_size
    test_steps_per_epoch = (num_test_sequences_per_epoch * seq_len_task) // batch_size

    train_dataset = StatefulOperationDataset(num_train_sequences_per_epoch, seq_len_task, noise_level_data, fixed_bit_A=fixed_bit_A_for_XOR_mode)
    test_dataset = StatefulOperationDataset(num_test_sequences_per_epoch, seq_len_task, noise_level_data, fixed_bit_A=fixed_bit_A_for_XOR_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    dpp_layer_shared_g = DPPLayer_SharedG(input_size, dpp_units, shared_g_dim_config)
    model_to_train = DPPModelBase(dpp_layer_shared_g, dpp_units, output_size).to(DEVICE)

    print(f"\nModellparameter: {count_parameters(model_to_train)}")

    optimizer = optim.Adam(model_to_train.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * train_steps_per_epoch, eta_min=1e-7)


    print(f"\n--- Training DPP SharedG (H={dpp_units}, SG_dim={shared_g_dim_config}) für Zustandsgesteuerte Operation ---")
    history, total_time, epoch_target, time_target = train_model_amp(
        model_to_train, train_loader, criterion, optimizer, scaler,
        epochs=epochs, model_name="DPP_StatefulOp", target_accuracy=target_accuracy_threshold,
        steps_per_epoch=train_steps_per_epoch
    )

    print("\n--- Finale Evaluation (Zustandsgesteuerte Operation) ---")
    final_acc, final_loss = evaluate_model_amp(model_to_train, test_loader, criterion, model_name="DPP_StatefulOp", steps_per_epoch=test_steps_per_epoch)
    print(f"DPP_StatefulOp - Parameter: {count_parameters(model_to_train)}")
    print(f"  Final Training Accuracy (Ende letzter Epoche): {history['accuracy'][-1]:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Total Training Time: {total_time:.3f}s")
    if epoch_target:
        print(f"  Reached {target_accuracy_threshold*100:.1f}% Train Acc at Epoch: {epoch_target} in {time_target:.3f}s")
    else:
        print(f"  Did not reach {target_accuracy_threshold*100:.1f}% train accuracy within {epochs} epochs.")

    # Alpha-Inspektion
    print("\n--- Alpha-Inspektion für einige Test-Samples (Zustandsgesteuerte Operation) ---")
    model_to_train.eval()
    test_iter = iter(test_loader)
    sample_count_inspect = 0
    max_inspect = 20 # Zeige mehr Samples
    print(f"Format Input: [xt_noisy, yt_minus_1_noisy]")
    print(f"Logik: y_t = (xt XOR {fixed_bit_A_for_XOR_mode}) if yt-1=0, else xt")
    print("Annahme für Alpha-Interpretation: Pfad A lernt (xt XOR const_A), Pfad B lernt xt.")
    print("  Wenn yt-1=0 (XOR-Modus), sollte Alpha tendenziell zu 1 gehen (um Pfad A zu wählen).")
    print("  Wenn yt-1=1 (Durchleit-Modus), sollte Alpha tendenziell zu 0 gehen (um Pfad B zu wählen).")
    print("----------------------------------------------------------------------------------")
    with torch.no_grad():
        while sample_count_inspect < max_inspect:
            try:
                batch_inputs, batch_labels = next(test_iter)
                batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
                _ = model_to_train(batch_inputs, return_alpha_flag=True)
                alphas_batch = model_to_train.last_alphas
                if alphas_batch is None: # Sicherheitscheck
                    print("Konnte Alphas nicht abrufen.")
                    break

                for i in range(batch_inputs.size(0)):
                    if sample_count_inspect >= max_inspect: break
                    inp = batch_inputs[i].cpu().numpy()
                    lbl = batch_labels[i].cpu().item()
                    alpha_vals = alphas_batch[i].cpu().numpy()
                    mean_alpha = np.mean(alpha_vals)
                    xt_approx = round(inp[0])
                    yt_minus_1_approx = round(inp[1])
                    print(f"Sample {sample_count_inspect+1}: Input~[{xt_approx}, {yt_minus_1_approx}], Target: {lbl:.0f}, Mean Alpha: {mean_alpha:.3f}, Alphas: {np.round(alpha_vals,2)}")
                    sample_count_inspect += 1
            except StopIteration:
                break
    if sample_count_inspect == 0:
        print("Keine Samples im Testloader gefunden für Inspektion.")