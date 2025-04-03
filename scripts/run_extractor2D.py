import sys
import os

# Añadir la raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

##########################################
# IMPORTAMOS TUS CLASES
##########################################
from src.ecgi.ECGiDataset import ECGiDataset
from src.ecgi.ECGMetricsComparator import ECGMetricsComparator
from src.models.cnn_models import RawCNNBase

from src.ecgi.ECGiTransformManager import ECGiTransformManager
from src.ecgi.ECGiTransformVisualizer import ECGiTransformVisualizer

##########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Ruta al archivo .mat (modificar según corresponda)
mat_file_path = '/workspace/pytorch_project/data/auckland/raw/auckland_data.mat'
ecgi_dataset = ECGiDataset(mat_file_path, device=device)
ecgi_dataset.print_summary()

# Extraer tensores en forma (B, T, C)
ecg_tensor = ecgi_dataset.ecg_tensor
emg_tensor = ecgi_dataset.emg_tensor

###############################################
# 1) Reordenar si tu transform requiere (B, C, T)
###############################################
# Suponiendo que tus datos están en (B, T, C) y
# quieres que la última dimensión sea "tiempo", 
# en este ejemplo DEJAMOS tal cual. 
# Si tu transform asume (B, C, T) => data_3d.permute(0,2,1).
# Revisa si lo necesitas:
emg_tensor_BCT = emg_tensor  # (B, T, C) – Ajustar si es necesario
# Mover a CPU para transform
emg_tensor_BCT = emg_tensor_BCT.cpu()

# Carpertas
transform_dir = "/workspace/pytorch_project/data/auckland/transform"
dataset_name  = ""

###############################################
# 2) Instanciar Manager y Visualizer
###############################################
manager = ECGiTransformManager(
    fs=1000,
    transform_dir=transform_dir,
    dataset_name=dataset_name
)

visualizer = ECGiTransformVisualizer(
    transform_dir=transform_dir,
    dataset_name=dataset_name,
    fs=1000
)

###############################################
# 3) Lista de transformaciones a aplicar
###############################################
transforms_list = [
    {"method": "rpm"},
    {"method": "gaf",  "gaf_mode": "summation"},
    {"method": "gaf",  "gaf_mode": "difference"},
    {"method": "mtf",       "n_bins": 8},
    {"method": "rp",        "threshold": 0.2, "metric": "euclidean"},
    {"method": "jrp",       "embedding_dim": 3, "time_delay": 1, "threshold": 0.1},
    {"method": "spectrogram","nperseg":32, "noverlap":16},
    {"method": "cwt",       "wavelet_type":"morlet", "scales":16}, 
    {"method": "dwt",       "wavelet":"db4", "level":2},
    {"method": "stft",      "window_type":"hann", "nperseg":64, "noverlap":32}
]

###############################################
# 4) Aplicar cada transformación con un PAA
#    factor, por ejemplo 2.
###############################################
paa_factor = 2

for transform_info in transforms_list:
    method = transform_info["method"]
    # Cada transform se guarda con un prefijo
    filename_prefix = f"auckland_emg_{method}"

    # Llamar a manager.apply_and_save_transform
    out_path = manager.apply_and_save_transform(
        data_3d=emg_tensor_BCT,    # (B, T, C) => si tu "TimeSeriesTo2DTransform" necesita (B, C, T) ajusta con permute
        transform_type=method,
        filename_prefix=filename_prefix,
        paa_factor=paa_factor,
        **{k:v for k,v in transform_info.items() if k != "method"}  # pasar kwargs extra, ej. gaf_mode, n_bins...
    )
    print(f"[OK] Transform: {method}, guardado en {out_path}")

    # 5) Cargamos el resultado y graficamos
    npy_name = f"{filename_prefix}_{method}.npy"
    data_4d = visualizer.load_transform_npy(npy_name)
    print(f"{method} => shape: {data_4d.shape}")

    # 6) Graficamos un canal
    # Seleccionamos (batch=0, electrode=0) por ejemplo
    visualizer.visualize_transform(
        transform_type=method,
        data_4d=data_4d,
        signal_index=0,
        electrode_index=0
    )

    # Podrías también graficar subplots 
    # (ej. 6 canales) si te interesa:
    # out_png = f"{filename_prefix}_{method}.png"
    # visualizer.visualize_transform_subplots(
    #     transform_type=method,
    #     data_4d=data_4d,
    #     signal_index=0,
    #     num_to_plot=6,
    #     num_cols=3,
    #     save_path=os.path.join(transform_dir, out_png)
    # )

print("\n=== ¡Listo! Se aplicaron todas las transformaciones ===")

##############################################################################
# (OPCIONAL) Entrenamiento con la salida 2D
##############################################################################
# ... Si tu modelo es 1D, no lo necesitas ...
