# src/data/datasets.py
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.ecgi.ECGiDataset import ECGiDataset

logger = logging.getLogger(__name__)

class ECGDataLoaderFactory:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @staticmethod
    def paa_time_reduction(signal: torch.Tensor, target_timesteps: int) -> torch.Tensor:
        B, C, T = signal.shape
        if target_timesteps >= T:
            return signal

        reduced = torch.zeros((B, C, target_timesteps), device=signal.device, dtype=signal.dtype)
        block_size = T / target_timesteps

        for m in range(target_timesteps):
            start = int(m * block_size)
            end = int((m+1) * block_size)
            end = min(end, T)
            reduced[:, :, m] = signal[:, :, start:end].mean(dim=-1)

        return reduced

    def create_loaders(self):
        """Crea DataLoaders para entrenamiento, validación y prueba"""
        try:
            # 1. Cargar dataset completo
            dataset = ECGiDataset(
                mat_file_path=self.config.data_path,
                device=self.device
            )
            
            # 2. Aplicar reducción PAA
            ecg_reduced = self.paa_time_reduction(dataset.ecg_tensor, self.config.time_steps)
            emg_reduced = self.paa_time_reduction(dataset.emg_tensor, self.config.time_steps)

            # Añadir en create_loaders():
            print("Forma de ECG después de PAA:", ecg_reduced.shape)  # Debe ser [N, C, T]
            print("Forma de EMG después de PAA:", emg_reduced.shape)  # Debe ser [N, C, T]

            # 3. Crear dataset completo
            full_dataset = TensorDataset(ecg_reduced.cpu(), emg_reduced.cpu())
            
            # 4. Dividir datos (70% train, 30% test)
            train_idx, val_idx = train_test_split(
                np.arange(len(full_dataset)),
                test_size=0.3,  # 30% validación
                random_state=42,
                shuffle=True  # Mejor práctica para mezclar datos
            )
    

            # 5. Crear DataLoaders
            train_loader = DataLoader(
                dataset=torch.utils.data.Subset(full_dataset, train_idx),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                dataset=torch.utils.data.Subset(full_dataset, val_idx),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            logger.info(f"""
                Loaders creados:
                - Train: {len(train_loader.dataset)} muestras
                - Val: {len(val_loader.dataset)} muestras
            """)

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Error creando DataLoaders: {str(e)}")
            raise