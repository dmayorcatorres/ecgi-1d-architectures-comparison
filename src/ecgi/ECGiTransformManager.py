# ====================================================
# src/ecgi/ECGTransformManager.py
# ====================================================
import os
import numpy as np
import torch
from src.ecgi.TimeSeriesTo2DTransform import TimeSeriesTo2DTransform
from typing import Union

class ECGiTransformManager:
    def __init__(self,
                 fs: int = 1000,
                 transform_dir: str = "transform",
                 dataset_name: str = "default_dataset"):
        """
        Clase que gestiona:
         1) Aplicar transformaciones 2D a datos 3D.
         2) Guardar los resultados en .npy.

        Args:
          fs: frecuencia de muestreo (ej. 1000 Hz).
          transform_dir: carpeta raíz donde se guardarán los resultados.
          dataset_name: nombre del dataset (subcarpeta dentro de transform_dir).
        """
        self.fs = fs
        self.transformer = TimeSeriesTo2DTransform(fs=fs)
        self.transform_dir = transform_dir
        self.dataset_name = dataset_name

        # Crear carpeta transform/<dataset_name> si no existe
        self.dataset_path = os.path.join(self.transform_dir, self.dataset_name)
        os.makedirs(self.dataset_path, exist_ok=True)

    def apply_and_save_transform(self,
                                 data_3d: Union[np.ndarray, torch.Tensor],
                                 transform_type: str,
                                 filename_prefix: str = "",
                                 paa_factor: int = 1,
                                 **kwargs) -> str:
        """
        Aplica la transformación 'transform_type' usando TimeSeriesTo2DTransform,
        y guarda el resultado en formato .npy.

        Args:
          data_3d: datos con forma (B, C, T) o (B, T, C) [siempre reordenar].
          transform_type: uno de ['rpm','gaf','mtf','rp','jrp','spectrogram','cwt','dwt','stft'].
          filename_prefix: un prefijo opcional para el nombre de salida.
          paa_factor: factor de PAA (Piecewise Aggregate Approximation). 
                      Si > 1, reduce la longitud temporal T → T//paa_factor.
          kwargs: parámetros específicos de la transformación (e.g. 'gaf_mode' para GAF).

        Returns:
          La ruta donde se guardó el archivo .npy
        """
        # Convertir a tensor si es np.ndarray
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()

        # Asegurarnos de que la clase transformadora reconoce el método
        if transform_type not in self.transformer.transformations:
            raise ValueError(f"Transform '{transform_type}' no soportada.")

        # IMPORTANTE: si tus datos vienen en forma (B, T, C),
        # reordenarlos a (B, C, T) antes de la transformación, e.g.:
        # data_3d = data_3d.permute(0, 2, 1)

        # Aplicar la transformación
        result_4d = self.transformer.transform(
            x=data_3d,
            method=transform_type,
            paa_factor=paa_factor,     # <- El factor PAA se usa aquí
            return_numpy=False,
            **kwargs
        )  # => Tensor shape depende del método: (B, C, dim1, dim2) o (B, dim1, dim2, C)

        # Convertir a np.ndarray y guardar
        result_np = result_4d.cpu().numpy()
        # Determinar nombre
        file_suffix = f"_{transform_type}" if filename_prefix else transform_type
        filename = f"{filename_prefix}{file_suffix}.npy" if filename_prefix else f"{transform_type}.npy"
        out_path = os.path.join(self.dataset_path, filename)

        np.save(out_path, result_np)
        return out_path