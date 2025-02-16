import sys
import os

# Añadir la raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Optional, Union, List
import logging
import pandas as pd
import optuna

from src.data.ECGiDataset import ECGiDataset
from src.utils.ECGMetricsComparator import ECGMetricsComparator
from src.models.cnn_models import RawCNNBase

# Configuración de datos y dispositivo
B, T = 8, 272                # n_signals y n_time
input_channels = 502         # Canales de EMG
output_channels = 163        # Canales de ECG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Ruta al archivo .mat (modificar según corresponda)
mat_file_path = '/workspace/pytorch_project/data/raw/kitpvc/kitpvc_data.mat'
ecgi_dataset = ECGiDataset(mat_file_path, device=device)
ecgi_dataset.print_summary()

# Obtener tensores (se asume forma: (n_signals, n_time, n_channels))
ecg_tensor = ecgi_dataset.get_ecg_tensor()  # Ej: (8, 272, 163)
emg_tensor = ecgi_dataset.get_emg_tensor()  # Ej: (8, 272, 502)

# Crear DataLoaders para entrenamiento y validación
batch_size = 2
train_dataset = TensorDataset(emg_tensor, ecg_tensor)
val_dataset = TensorDataset(emg_tensor, ecg_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lista de variantes a optimizar
variants = ["simple", "fcn", "resnet", "dilated", "disjoint", "multiscale", "mc-dcnn", "mc-cnn", "res-cnn"]

#############################################
# Función objetivo para Optuna (solo hiperparámetros generales)
#############################################
def objective(trial, variant):
    # Hiperparámetro general: tasa de aprendizaje
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Instanciar el modelo usando la misma interfaz para todas las variantes
    model = RawCNNBase(variant=variant, time_steps=T,
                        input_channels=input_channels,
                        output_channels=output_channels)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Entrenar por 10 épocas para evaluación rápida
    n_epochs = 10
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)

    # Evaluación en validación
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    comparator = ECGMetricsComparator(fs=1000)
    comparator.set_time_domain_signals(all_targets, all_predictions)
    cc_mean, _ = comparator.compute_time_cc()
    (_, _), (rmse_mean, _) = comparator.compute_time_mse_rmse()

    # Objetivo: minimizar (1 - CC_mean) + RMSE_mean
    return (1 - cc_mean) + rmse_mean

#############################################
# Optimización con Optuna para cada variante
#############################################
best_results = {}
for var in variants:
    print(f"\n=== Optimización para variante: {var} ===")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, variant=var), n_trials=30)
    best_trial = study.best_trial
    best_results[var] = {
        'best_value': best_trial.value,
        'params': best_trial.params
    }
    print(f"Mejor trial para {var}: Value = {best_trial.value}, Params = {best_trial.params}")

#############################################
# Reentrenamiento con los mejores hiperparámetros (solo lr se usa) y cálculo de métricas finales
#############################################
final_results = {}
num_epochs_full = 50

for var in variants:
    print(f"\n=== Reentrenando variante: {var} con mejores parámetros ===")
    # Extraer lr del mejor trial
    lr = best_results[var]['params'].get("lr", 1e-3)

    # Instanciar el modelo sin pasar hiperparámetros adicionales
    model = RawCNNBase(variant=var, time_steps=T, input_channels=input_channels,
                        output_channels=output_channels)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_history = []
    val_history = []

    for epoch in range(num_epochs_full):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_history.append(epoch_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_history.append(epoch_val_loss)
        print(f"Epoch {epoch+1:02d}: Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Calcular métricas finales
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    comparator = ECGMetricsComparator(fs=1000)
    comparator.set_time_domain_signals(all_targets, all_predictions)
    mae_mean, mae_std = comparator.compute_time_mae()
    cc_mean, cc_std = comparator.compute_time_cc()
    (mse_mean, mse_std), (rmse_mean, rmse_std) = comparator.compute_time_mse_rmse()
    snr_mean, snr_std = comparator.compute_time_snr()

    final_results[var] = {
        'Val Loss': val_history[-1],
        'MAE_mean': mae_mean,
        'MAE_std': mae_std,
        'CC_mean': cc_mean,
        'CC_std': cc_std,
        'MSE_mean': mse_mean,
        'MSE_std': mse_std,
        'RMSE_mean': rmse_mean,
        'RMSE_std': rmse_std,
        'SNR_mean': snr_mean,
        'SNR_std': snr_std,
        'Best Params': best_results[var]['params'],
        'train_loss_history': train_history,
        'val_loss_history': val_history
    }
    print(f"\nVariante {var} - Métricas Finales:")
    print(f"Val Loss: {val_history[-1]:.4f}")
    print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"CC:  {cc_mean:.4f} ± {cc_std:.4f}")
    print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"SNR: {snr_mean:.4f} ± {snr_std:.4f}")

#############################################
# Tabla comparativa final
#############################################
rows = []
for var, metrics in final_results.items():
    row = {
        'Variante': var,
        'Val Loss': metrics['Val Loss'],
        'MAE (mean)': metrics['MAE_mean'],
        'MAE (std)': metrics['MAE_std'],
        'CC (mean)': metrics['CC_mean'],
        'CC (std)': metrics['CC_std'],
        'MSE (mean)': metrics['MSE_mean'],
        'MSE (std)': metrics['MSE_std'],
        'RMSE (mean)': metrics['RMSE_mean'],
        'RMSE (std)': metrics['RMSE_std'],
        'SNR (mean)': metrics['SNR_mean'],
        'SNR (std)': metrics['SNR_std'],
        'Best Params': metrics['Best Params']
    }
    rows.append(row)
df = pd.DataFrame(rows)
print("\n=== Tabla Comparativa de Resultados Finales ===")
print(df)

# Opcional: Visualización de curvas de pérdida para la variante considerada mejor (por ejemplo, "resnet")
best_variant = "resnet"
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(final_results[best_variant]['train_loss_history'], label="Train Loss")
plt.title(f'Training Loss - {best_variant}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_results[best_variant]['val_loss_history'], label="Val Loss")
plt.title(f'Validation Loss - {best_variant}')
plt.legend()

plt.tight_layout()
plt.show()
