import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import optuna

# --- Seed-Initialisierung (EINMALIG GLOBAL) ---
SEED = 47
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"INFO: Globaler Seed wurde auf {seed_value} gesetzt.")

set_seed(SEED)

# Optional: Matplotlib für die Visualisierung
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib nicht gefunden. Die Visualisierung wird übersprungen.")

# --- NEUER DPPResidualVFNConvLayer ---
class DPPResidualVFNConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation_fn=nn.ReLU(), dropout_rate=0.0, shared_g_dim_factor=0.25):
        super().__init__()
        self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        self.conv_res_A = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_res_B = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        internal_g_channels = max(1, int(in_channels * shared_g_dim_factor))
        if internal_g_channels == 0 and in_channels > 0: # Fallback
             internal_g_channels = 1

        self.conv_gate_shared_features = nn.Conv2d(in_channels, internal_g_channels, kernel_size=1, bias=False)
        self.gate_feature_norm = nn.BatchNorm2d(internal_g_channels)
        self.conv_gate_alpha_projection = nn.Conv2d(internal_g_channels, out_channels, kernel_size=1)

        self.activation_res = activation_fn
        self.norm_final = nn.BatchNorm2d(out_channels)
        self.dropout_final = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self._initialize_layer_weights()

    def _initialize_layer_weights(self):
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if isinstance(self.activation_res, nn.ReLU) else 'leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        main_path_out = self.conv_main(x)
        res_A_out = self.conv_res_A(x)
        res_B_out = self.conv_res_B(x)

        gate_shared_feat = self.conv_gate_shared_features(x)
        gate_shared_feat = self.gate_feature_norm(F.relu(gate_shared_feat))
        alpha_logits = self.conv_gate_alpha_projection(gate_shared_feat)
        alpha = torch.sigmoid(alpha_logits)

        mixed_residual = alpha * res_A_out + (1 - alpha) * res_B_out
        activated_residual = self.activation_res(mixed_residual)

        combined = main_path_out + activated_residual
        normalized = self.norm_final(combined)
        output = self.dropout_final(normalized)
        return output

# --- NEUES DPP-VFN-CNN Modell für CIFAR-10 ---
class DPP_VFNCIFAR10Net(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, dropout_conv=0.1, dropout_fc=0.5,
                 shared_g_dim_factor=0.25, activation_choice="ReLU"):
        super().__init__()

        if activation_choice == "ReLU":
            conv_activation_fn = nn.ReLU()
        elif activation_choice == "GELU":
            conv_activation_fn = nn.GELU()
        elif activation_choice == "SiLU":
            conv_activation_fn = nn.SiLU()
        else:
            conv_activation_fn = nn.ReLU() # Default

        self.block1_layer1 = DPPResidualVFNConvLayer(3, base_channels, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.block1_layer2 = DPPResidualVFNConvLayer(base_channels, base_channels, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_layer1 = DPPResidualVFNConvLayer(base_channels, base_channels * 2, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.block2_layer2 = DPPResidualVFNConvLayer(base_channels * 2, base_channels * 2, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_layer1 = DPPResidualVFNConvLayer(base_channels * 2, base_channels * 4, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.block3_layer2 = DPPResidualVFNConvLayer(base_channels * 4, base_channels * 4, kernel_size=3, padding=1,
                                                     dropout_rate=dropout_conv, activation_fn=conv_activation_fn,
                                                     shared_g_dim_factor=shared_g_dim_factor)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(base_channels * 4, num_classes)

        self._initialize_fc_weights()

    def _initialize_fc_weights(self):
        if isinstance(self.fc, nn.Linear):
            nn.init.normal_(self.fc.weight, 0, 0.01)
            nn.init.constant_(self.fc.bias, 0)
            # Entferne das print hier, um die Konsole sauberer zu halten während HPO
            # print("INFO: FC Layer Gewichte initialisiert.")

    def forward(self, x):
        x = self.pool1(self.block1_layer2(self.block1_layer1(x)))
        x = self.pool2(self.block2_layer2(self.block2_layer1(x)))
        x = self.pool3(self.block3_layer2(self.block3_layer1(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

# --- Datenvorbereitung für CIFAR-10 mit erweiterter Augmentierung (aus Test8) ---
def get_cifar10_dataloaders(batch_size=128, num_workers=2, use_advanced_augmentations=True, download_data=True): # Neuer Parameter download_data
    # Entferne das print hier, um die Konsole sauberer zu halten während HPO
    # print(f"Lade CIFAR-10 Daten... Erweiterte Augmentierung: {use_advanced_augmentations}")
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    if use_advanced_augmentations:
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False),
            normalize,
        ]
    else:
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=download_data, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=download_data, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Entferne das print hier
    # print("CIFAR-10 Daten geladen.")
    return trainloader, testloader, classes

# --- Trainings- und Evaluierungsfunktion (angepasst für Optuna, finales Training UND AMP) ---
def train_eval_loop(model, trainloader, valloader, criterion, optimizer, scheduler, num_epochs, device,
                    scaler,
                    trial=None, current_trial_num=None, non_blocking_transfer=True):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc_for_this_run = 0.0
    use_amp_local = scaler is not None and device.type == 'cuda'

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=non_blocking_transfer)
            labels = labels.to(device, non_blocking=non_blocking_transfer)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp_local):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if use_amp_local:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(trainloader)
        history['train_loss'].append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        model.eval()
        running_val_loss = 0.0; correct_val = 0; total_val = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device, non_blocking=non_blocking_transfer)
                labels = labels.to(device, non_blocking=non_blocking_transfer)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp_local):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0); correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(valloader)
        val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(avg_val_loss); history['val_acc'].append(val_accuracy)

        if val_accuracy > best_val_acc_for_this_run:
            best_val_acc_for_this_run = val_accuracy

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()


        epoch_duration = time.time() - epoch_start_time
        log_prefix = f"Optuna Trial {current_trial_num}, " if trial else ""
        # Reduziere Output-Frequenz für HPO, logge nur alle paar Epochen oder wenn eine Verbesserung erzielt wird.
        # Hier loggen wir weiterhin jede Epoche, aber man könnte es anpassen.
        print(f"{log_prefix}Epoche [{epoch+1}/{num_epochs}], TL: {avg_train_loss:.4f}, VL: {avg_val_loss:.4f}, "
              f"VA: {val_accuracy:.2f}%, LR: {current_lr:.1e}, Dauer: {epoch_duration:.2f}s")

        if trial:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                print(f"  Trial {current_trial_num} pruned at epoch {epoch+1}.")
                raise optuna.exceptions.TrialPruned()

    return best_val_acc_for_this_run, history


# --- Plotting-Funktion (aus Test8) ---
def plot_training_history(history, title='Trainingsverlauf'):
    if not MATPLOTLIB_AVAILABLE: return
    fig, ax1 = plt.subplots(figsize=(10, 4))
    color = 'tab:red'; ax1.set_xlabel('Epoche'); ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, ls='-', label='Trainings-Loss')
    ax1.plot(history['val_loss'], color=color, ls='--', label='Validierungs-Loss')
    ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='center left'); ax1.grid(True, ls=':', alpha=0.7)
    ax2 = ax1.twinx(); color = 'tab:blue'; ax2.set_ylabel('Genauigkeit (%)', color=color)
    ax2.plot(history['val_acc'], color=color, ls='-', label='Validierungs-Genauigkeit')
    ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='center right')
    fig.tight_layout(); plt.title(title); plt.show()

# --- Globale Konstanten und Optuna Objective Funktion ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()
NUM_EPOCHS_PER_TRIAL = 25
N_OPTUNA_TRIALS = 50
NUM_EPOCHS_FINAL_TRAINING = 100
USE_AMP_GLOBAL = torch.cuda.is_available()

HPO_BATCH_SIZE = 256
# ÄNDERUNG: num_workers für HPO auf 0 setzen, um mehrfache Ausgaben zu reduzieren
NUM_WORKERS_DATALOADER_HPO = 0
NUM_WORKERS_DATALOADER_FINAL = 4 if DEVICE.type == 'cuda' else 2

print("Einmaliges globales Laden der Daten für HPO...")
TRAINLOADER_HPO, TESTLOADER_HPO, CLASSES_CIFAR10_HPO = get_cifar10_dataloaders(
    batch_size=HPO_BATCH_SIZE, use_advanced_augmentations=True, num_workers=NUM_WORKERS_DATALOADER_HPO, download_data=True # Erster Download
)
print("Daten für HPO geladen.")


def objective(trial: optuna.trial.Trial) -> float:
    # Seed pro Trial, um Reproduzierbarkeit innerhalb des Trials zu gewährleisten
    # und Variabilität zwischen den Trials sicherzustellen.
    # Die print-Ausgabe von set_seed hier entfernen, um Konsole sauberer zu halten.
    current_seed = SEED + trial.number
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    np.random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)


    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGDNesterov"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    dropout_conv = trial.suggest_float("dropout_conv", 0.0, 0.3)
    dropout_fc = trial.suggest_float("dropout_fc", 0.2, 0.6)
    activation_choice = trial.suggest_categorical("activation_conv", ["ReLU", "GELU", "SiLU"])
    shared_g_dim_factor = trial.suggest_float("shared_g_dim_factor", 0.1, 0.75)

    model = DPP_VFNCIFAR10Net(
        num_classes=len(CLASSES_CIFAR10_HPO),
        base_channels=base_channels,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
        shared_g_dim_factor=shared_g_dim_factor,
        activation_choice=activation_choice
    ).to(DEVICE)

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"])
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_PER_TRIAL, eta_min=1e-7)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=NUM_EPOCHS_PER_TRIAL // 5, factor=0.2, min_lr=1e-7)

    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)


    print(f"Optuna Trial {trial.number}: LR={lr:.1e}, Optim={optimizer_name}, WD={weight_decay:.1e}, BC={base_channels}, Act={activation_choice}, DropC={dropout_conv:.2f}, DropFC={dropout_fc:.2f}, G_Factor={shared_g_dim_factor:.2f}, Sched={scheduler_name}")

    best_val_acc_for_trial, _ = train_eval_loop(
        model, TRAINLOADER_HPO, TESTLOADER_HPO, CRITERION, optimizer, scheduler,
        NUM_EPOCHS_PER_TRIAL, DEVICE, scaler, trial, trial.number
    )

    return best_val_acc_for_trial

# --- Hauptskript ---
if __name__ == "__main__":
    print(f"Verwende Gerät: {DEVICE}. AMP global aktiviert: {USE_AMP_GLOBAL}")
    if USE_AMP_GLOBAL:
        print("INFO: Automatic Mixed Precision (AMP) wird für Training und Evaluation verwendet.")

    print("\nStarte Hyperparameter-Optimierung mit Optuna...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=7, n_warmup_steps=NUM_EPOCHS_PER_TRIAL // 3, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    timeout_hpo = 3600 * 2
    try:
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=timeout_hpo)
    except KeyboardInterrupt:
        print("Optuna-Optimierung manuell abgebrochen.")
    except Exception as e:
        print(f"Fehler während der Optuna-Optimierung: {e}")
        import traceback
        traceback.print_exc()


    print("\nHyperparameter-Optimierung abgeschlossen.")
    best_params = {}
    if study.best_trial:
        best_trial = study.best_trial
        print("Bester Trial:")
        print(f"  Wert (Beste Validierungs-Genauigkeit im Trial): {best_trial.value:.4f}%")
        print("  Parameter: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        best_params = best_trial.params
    else:
        print("Kein bester Trial gefunden. Verwende Standardparameter für finales Training.")
        best_params = {
            "lr": 1e-3, "optimizer": "AdamW", "weight_decay": 1e-4,
            "base_channels": 48, "dropout_conv": 0.1, "dropout_fc": 0.4,
            "activation_conv": "ReLU", "shared_g_dim_factor": 0.25,
            "scheduler": "CosineAnnealingLR"
        }

    print(f"\nStarte finales Training mit den besten/Standard-Hyperparametern für {NUM_EPOCHS_FINAL_TRAINING} Epochen...")
    # Setze Seed für finales Training, um Reproduzierbarkeit des finalen Laufs zu gewährleisten
    final_seed = SEED + 1000
    torch.manual_seed(final_seed)
    random.seed(final_seed)
    np.random.seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)
        torch.backends.cudnn.benchmark = True # Aktiviere für Performance im finalen Training
        print("INFO: torch.backends.cudnn.benchmark = True für finales Training gesetzt.")


    final_model = DPP_VFNCIFAR10Net(
        num_classes=len(CLASSES_CIFAR10_HPO),
        base_channels=best_params.get("base_channels", 48),
        dropout_conv=best_params.get("dropout_conv", 0.1),
        dropout_fc=best_params.get("dropout_fc", 0.4),
        shared_g_dim_factor=best_params.get("shared_g_dim_factor", 0.25),
        activation_choice=best_params.get("activation_conv", "ReLU")
    ).to(DEVICE)

    final_optimizer_name = best_params.get("optimizer", "AdamW")
    final_lr = best_params.get("lr", 1e-3)
    final_wd = best_params.get("weight_decay", 1e-4)

    if final_optimizer_name == "AdamW":
        final_optimizer = optim.AdamW(final_model.parameters(), lr=final_lr, weight_decay=final_wd)
    else:
        final_optimizer = optim.SGD(final_model.parameters(), lr=final_lr, momentum=0.9, weight_decay=final_wd, nesterov=True)

    final_scheduler_name = best_params.get("scheduler", "CosineAnnealingLR")
    if final_scheduler_name == "CosineAnnealingLR":
        final_scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=NUM_EPOCHS_FINAL_TRAINING, eta_min=1e-7)
    else:
        final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, 'max', patience=NUM_EPOCHS_FINAL_TRAINING // 10, factor=0.2, min_lr=1e-7)

    # Datenlader für das finale Training (mit potenziell mehr Workern)
    print("Laden der Daten für das finale Training...")
    final_trainloader, final_testloader, final_classes = get_cifar10_dataloaders(
        batch_size=HPO_BATCH_SIZE, use_advanced_augmentations=True, num_workers=NUM_WORKERS_DATALOADER_FINAL, download_data=False # Daten sollten schon da sein
    )
    print("Daten für finales Training geladen.")


    final_scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)

    print(f"Finale Trainingsparameter: LR={final_lr:.1e}, Optim={final_optimizer_name}, WD={final_wd:.1e}, Sched={final_scheduler_name}, "
          f"BC={best_params.get('base_channels', 48)}, Act={best_params.get('activation_conv', 'ReLU')}, G_Factor={best_params.get('shared_g_dim_factor', 0.25)}")

    _, final_history = train_eval_loop(
        final_model, final_trainloader, final_testloader, CRITERION, final_optimizer, final_scheduler,
        NUM_EPOCHS_FINAL_TRAINING, DEVICE, final_scaler
    )

    if MATPLOTLIB_AVAILABLE:
        plot_training_history(final_history, title=f'Finales Training DPP-VFN-CNN auf CIFAR-10 (Beste HPs: VA {max(final_history.get("val_acc", [0])) :.2f}%)')

    print("\nFinale Evaluierung des optimierten Modells auf dem Testset:")
    final_model.eval()
    correct_final = 0; total_final = 0; running_test_loss = 0.0
    non_blocking_final_eval = DEVICE.type == 'cuda'

    with torch.no_grad():
        for images, labels in final_testloader:
            images = images.to(DEVICE, non_blocking=non_blocking_final_eval)
            labels = labels.to(DEVICE, non_blocking=non_blocking_final_eval)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP_GLOBAL):
                outputs = final_model(images)
                loss = CRITERION(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_final += labels.size(0)
            correct_final += (predicted == labels).sum().item()

    final_test_accuracy = 100 * correct_final / total_final
    avg_test_loss = running_test_loss / len(final_testloader)
    print(f'Endgültige Genauigkeit des optimierten Netzwerks auf den 10000 Testbildern: {final_test_accuracy:.2f} %')
    print(f'Durchschnittlicher Test-Loss: {avg_test_loss:.4f}')

    # torch.save(final_model.state_dict(), "dpp_vfn_cifar10_final_model.pth")
    # print("Finales Modell gespeichert als dpp_vfn_cifar10_final_model.pth")