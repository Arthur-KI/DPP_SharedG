import torch
import torch.nn as nn
import torch.nn.functional as F # Für F.relu in DPPResidualVFNConvLayer
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import time
import matplotlib.pyplot as plt
import numpy as np
# Optuna wird nicht mehr benötigt für diesen reinen Trainingslauf

# --- Seed-Initialisierung (EINMALIG GLOBAL) ---
SEED = 47 # Sie können diesen Seed für Reproduzierbarkeit beibehalten oder ändern
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"INFO: Globaler Seed wurde auf {seed_value} gesetzt.")

# Optional: Matplotlib für die Visualisierung
# Diese Variable muss global definiert sein, damit sie im __main__ Block verfügbar ist.
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
            print("INFO: FC Layer Gewichte initialisiert.")

    def forward(self, x):
        x = self.pool1(self.block1_layer2(self.block1_layer1(x)))
        x = self.pool2(self.block2_layer2(self.block2_layer1(x)))
        x = self.pool3(self.block3_layer2(self.block3_layer1(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

# --- Datenvorbereitung für CIFAR-10 mit erweiterter Augmentierung ---
def get_cifar10_dataloaders(batch_size=128, num_workers=2, use_advanced_augmentations=True, download_data=True):
    print(f"Datenvorbereitung: CIFAR-10 Daten. Erweiterte Augmentierung: {use_advanced_augmentations}, Download: {download_data}")
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
    print("Datenvorbereitung abgeschlossen.")
    return trainloader, testloader, classes

# --- Trainings- und Evaluierungsfunktion mit AMP ---
def train_eval_loop(model, trainloader, valloader, criterion, optimizer, scheduler, num_epochs, device,
                    scaler, non_blocking_transfer=True):
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
            # Optional: Modell hier speichern, wenn es das beste bisherige ist
            # torch.save(model.state_dict(), f"dpp_vfn_epoch_{epoch+1}_va_{val_accuracy:.2f}.pth")


        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoche [{epoch+1}/{num_epochs}], TL: {avg_train_loss:.4f}, VL: {avg_val_loss:.4f}, "
              f"VA: {val_accuracy:.2f}%, LR: {current_lr:.1e}, Dauer: {epoch_duration:.2f}s")

    return best_val_acc_for_this_run, history


# --- Plotting-Funktion ---
def plot_training_history(history, title='Trainingsverlauf'):
    if not MATPLOTLIB_AVAILABLE: # Prüfe die global definierte Variable
        print("Matplotlib nicht verfügbar. Plotting übersprungen.")
        return
    fig, ax1 = plt.subplots(figsize=(10, 4))
    color = 'tab:red'; ax1.set_xlabel('Epoche'); ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, ls='-', label='Trainings-Loss')
    ax1.plot(history['val_loss'], color=color, ls='--', label='Validierungs-Loss')
    ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='center left'); ax1.grid(True, ls=':', alpha=0.7)
    ax2 = ax1.twinx(); color = 'tab:blue'; ax2.set_ylabel('Genauigkeit (%)', color=color)
    ax2.plot(history['val_acc'], color=color, ls='-', label='Validierungs-Genauigkeit')
    ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='center right')
    fig.tight_layout(); plt.title(title); plt.show()

# --- Globale Konstanten ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()
NUM_EPOCHS_TRAINING = 110
USE_AMP_GLOBAL = torch.cuda.is_available()
BATCH_SIZE = 256
NUM_WORKERS_DATALOADER = 4 if DEVICE.type == 'cuda' else 0

# --- Hauptskript für den direkten Trainingslauf ---
if __name__ == "__main__":
    run_seed = SEED + 2000
    set_seed(run_seed)

    print(f"Verwende Gerät: {DEVICE}. AMP global aktiviert: {USE_AMP_GLOBAL}")
    if USE_AMP_GLOBAL:
        print("INFO: Automatic Mixed Precision (AMP) wird für Training und Evaluation verwendet.")
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("INFO: torch.backends.cudnn.benchmark = True für Training gesetzt.")

    best_params = {
        "lr": 0.002003627237023472,
        "optimizer": "AdamW",
        "weight_decay": 1.2138009911679477e-05,
        "base_channels": 64,
        "dropout_conv": 0.0013585700483887998,
        "dropout_fc": 0.4800340317125635,
        "activation_conv": "GELU",
        "shared_g_dim_factor": 0.6294179072415094,
        "scheduler": "ReduceLROnPlateau"
    }
    print("\nVerwendete Hyperparameter für diesen Lauf:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    model = DPP_VFNCIFAR10Net(
        num_classes=10,
        base_channels=best_params["base_channels"],
        dropout_conv=best_params["dropout_conv"],
        dropout_fc=best_params["dropout_fc"],
        shared_g_dim_factor=best_params["shared_g_dim_factor"],
        activation_choice=best_params["activation_conv"]
    ).to(DEVICE)

    if best_params["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9, weight_decay=best_params["weight_decay"], nesterov=True)

    if best_params["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_TRAINING, eta_min=1e-7)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=NUM_EPOCHS_TRAINING // 10, factor=0.2, min_lr=1e-7, verbose=True)

    trainloader, testloader, classes = get_cifar10_dataloaders(
        batch_size=BATCH_SIZE, use_advanced_augmentations=True, num_workers=NUM_WORKERS_DATALOADER, download_data=True
    )

    scaler = torch.amp.GradScaler(enabled=USE_AMP_GLOBAL)

    print(f"\nStarte Training für {NUM_EPOCHS_TRAINING} Epochen...")
    _, history = train_eval_loop(
        model, trainloader, testloader, CRITERION, optimizer, scheduler,
        NUM_EPOCHS_TRAINING, DEVICE, scaler
    )

    # Vor dem Plotten sicherstellen, dass MATPLOTLIB_AVAILABLE geprüft wird
    if MATPLOTLIB_AVAILABLE:
        plot_training_history(history, title=f'DPP-VFN-CNN auf CIFAR-10 (VA {max(history.get("val_acc", [0])) :.2f}%)')
    else:
        print("Plotting übersprungen, da Matplotlib nicht verfügbar ist.")


    print("\nFinale Evaluierung des Modells auf dem Testset:")
    model.eval()
    correct_final = 0; total_final = 0; running_test_loss = 0.0
    non_blocking_final_eval = DEVICE.type == 'cuda'

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE, non_blocking=non_blocking_final_eval)
            labels = labels.to(DEVICE, non_blocking=non_blocking_final_eval)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP_GLOBAL):
                outputs = model(images)
                loss = CRITERION(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_final += labels.size(0)
            correct_final += (predicted == labels).sum().item()

    final_test_accuracy = 100 * correct_final / total_final
    avg_test_loss = running_test_loss / len(testloader)
    print(f'Endgültige Genauigkeit des Netzwerks auf den 10000 Testbildern: {final_test_accuracy:.2f} %')
    print(f'Durchschnittlicher Test-Loss: {avg_test_loss:.4f}')

    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # model_save_path = f"dpp_vfn_cifar10_final_model_{timestamp}_va{final_test_accuracy:.2f}.pth"
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Finales Modell gespeichert als {model_save_path}")