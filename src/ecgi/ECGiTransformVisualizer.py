# ====================================================
# src/ecgi/ECGTransformVisualizer.py
# ====================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Union

from src.ecgi.TimeSeriesTo2DTransform import TimeSeriesTo2DTransform

class ECGiTransformVisualizer:
    def __init__(
        self,
        transform_dir: str = "transform",
        dataset_name: str = "default_dataset",
        fs: float = 1000
    ):
        """
        Clase que gestiona:
         - Transformar series 3D a 2D,
         - Guardar/ cargar tensores,
         - Visualizar los resultados.
         
        Args:
          transform_dir: carpeta raíz donde se guardarán/cargarán los .npy.
          dataset_name: subcarpeta (opcional).
          fs: frecuencia de muestreo que usará TimeSeriesTo2DTransform.
        """
        self.transform_dir = transform_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.transform_dir, self.dataset_name)
        os.makedirs(self.dataset_path, exist_ok=True)

        # Instancia la clase de transformaciones 2D
        self.transformer = TimeSeriesTo2DTransform(fs=fs)

    def transform_and_save_npy(
        self,
        data_3d: Union[np.ndarray, torch.Tensor],
        transform_type: str,
        filename_prefix: str = "",
        paa_factor: int = 1,
        return_numpy: bool = False,
        **kwargs
    ) -> str:
        """
        Aplica la transformación 'transform_type' (e.g. 'gaf','rpm','mtf', etc.),
        y guarda el resultado como .npy dentro de 'dataset_path'.

        data_3d => (B, C, T) si la clase TimeSeriesTo2DTransform lo requiere.

        Args:
        data_3d: tensores o arrays.
        transform_type: str con el nombre de la transformación (rpm, gaf, mtf...).
        filename_prefix: prefijo para el archivo .npy.
        paa_factor: factor PAA para reducir la longitud temporal. 
                    (T -> T//paa_factor)
        return_numpy: si True, se retorna np.ndarray en vez de torch.Tensor.
        kwargs: parámetros para la transformación (ej. gaf_mode='summation').

        Returns:
        out_path: ruta del archivo .npy guardado.
        """
        # Convertir a float CPU
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        else:
            data_3d = data_3d.float().cpu()

        # Aplicar la transformación
        result_4d = self.transformer.transform(
            x=data_3d,
            method=transform_type,
            paa_factor=paa_factor,   # <--- El factor PAA se usa aquí
            return_numpy=False,
            **kwargs
        )

        # Convertir a numpy y guardar
        result_np = result_4d.cpu().numpy()
        suffix = f"_{transform_type}" if filename_prefix else transform_type
        filename = f"{filename_prefix}{suffix}.npy" if filename_prefix else f"{transform_type}.npy"
        out_path = os.path.join(self.dataset_path, filename)
        np.save(out_path, result_np)

        return out_path


    def transform_and_save_npy_chunked(
        self,
        data_3d: Union[np.ndarray, torch.Tensor],
        transform_type: str,
        filename_prefix: str = "",
        chunk_size: int = 2,
        paa_factor: int = 1,
        **kwargs
    ) -> str:
        # Asegurar tensor en CPU
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        else:
            data_3d = data_3d.float().cpu()

        B, C, T = data_3d.shape
        results = []
        for start_idx in range(0, B, chunk_size):
            end_idx = start_idx + chunk_size
            chunk = data_3d[start_idx:end_idx]  
            chunk_transformed = self.transformer.transform(
                x=chunk, method=transform_type,
                paa_factor=paa_factor,
                return_numpy=False,
                **kwargs
            )
            results.append(chunk_transformed.cpu())

        full_result = torch.cat(results, dim=0)
        result_np = full_result.numpy()

        suffix = f"_{transform_type}" if filename_prefix else transform_type
        filename = f"{filename_prefix}{suffix}.npy" if filename_prefix else f"{transform_type}.npy"
        out_path = os.path.join(self.dataset_path, filename)
        np.save(out_path, result_np)

        return out_path
    
    def load_transform_npy(self, filename: str) -> np.ndarray:
        """
        Carga un archivo .npy (tensor 4D) desde dataset_path.
        """
        file_path = os.path.join(self.dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        data_4d = np.load(file_path)
        return data_4d

    # --------------------------------------------------------------------------
    # Visualización genérica de transformaciones 2D
    # --------------------------------------------------------------------------
    def visualize_transform(self,
                            transform_type: str,
                            data_4d: Union[np.ndarray, torch.Tensor],
                            signal_index: int = 0,
                            electrode_index: Optional[int] = None,
                            num_to_plot: Optional[int] = None,
                            save_path: Optional[str] = None):
        """
        Visualiza un tensor 4D resultante de una transformación 2D.
        data_4d => shape (B, C, dim1, dim2) o (B, dim1, dim2, C).
        """
        if isinstance(data_4d, torch.Tensor):
            data_4d = data_4d.cpu().numpy()

        shape_ = data_4d.shape
        B = shape_[0]
        if not (0 <= signal_index < B):
            print("signal_index fuera de rango.")
            return

        # Heurística para saber dónde está la dimensión de canales
        is_channel_in_dim1 = shape_[1] < 300

        if is_channel_in_dim1:
            # => (B, C, dim1, dim2)
            C = shape_[1]
            dim1 = shape_[2]
            dim2 = shape_[3]
            self._visualize_2dmap_b_c_dim1_dim2(
                transform_type, data_4d, signal_index, C, dim1, dim2,
                electrode_index, num_to_plot, save_path
            )
        else:
            # => (B, dim1, dim2, C)
            dim1 = shape_[1]
            dim2 = shape_[2]
            C = shape_[3]
            self._visualize_2dmap_b_dim1_dim2_c(
                transform_type, data_4d, signal_index, C, dim1, dim2,
                electrode_index, num_to_plot, save_path
            )

    def _visualize_2dmap_b_c_dim1_dim2(self,
                                       transform_type: str,
                                       data_4d: np.ndarray,
                                       signal_index: int,
                                       C: int,
                                       dim1: int,
                                       dim2: int,
                                       electrode_index: Optional[int],
                                       num_to_plot: Optional[int],
                                       save_path: Optional[str]):
        """
        Caso: data_4d => (B, C, dim1, dim2).
        """
        data_3d = data_4d[signal_index]  # => (C, dim1, dim2)

        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                print("electrode_index fuera de rango.")
                return
            arr_2d = data_3d[electrode_index]  # (dim1, dim2)
            self._plot_single_2d(transform_type, signal_index, electrode_index,
                                 arr_2d, save_path)
        elif num_to_plot is not None and num_to_plot > 0:
            step = max(1, C // num_to_plot)
            selected = list(range(0, C, step))[:num_to_plot]
            self._plot_multiple_2d(transform_type, signal_index, selected,
                                   data_3d, save_path)
        else:
            print("Debes especificar electrode_index o num_to_plot.")

    def _visualize_2dmap_b_dim1_dim2_c(self,
                                       transform_type: str,
                                       data_4d: np.ndarray,
                                       signal_index: int,
                                       C: int,
                                       dim1: int,
                                       dim2: int,
                                       electrode_index: Optional[int],
                                       num_to_plot: Optional[int],
                                       save_path: Optional[str]):
        """
        Caso: data_4d => (B, dim1, dim2, C).
        """
        data_3d = data_4d[signal_index]  # => (dim1, dim2, C)

        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                print("electrode_index fuera de rango.")
                return
            arr_2d = data_3d[:, :, electrode_index]  # (dim1, dim2)
            self._plot_single_2d(transform_type, signal_index, electrode_index,
                                 arr_2d, save_path)
        elif num_to_plot is not None and num_to_plot > 0:
            step = max(1, C // num_to_plot)
            selected = list(range(0, C, step))[:num_to_plot]
            self._plot_multiple_2d_alt(transform_type, signal_index, selected,
                                       data_3d, save_path)
        else:
            print("Debes especificar electrode_index o num_to_plot.")

    def _plot_single_2d(self,
                        transform_type: str,
                        signal_index: int,
                        electrode_index: int,
                        arr_2d: np.ndarray,
                        save_path: Optional[str]):
        """
        Grafica un único mapa 2D (dim1 x dim2).
        """
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(arr_2d, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{transform_type.upper()} - Sig {signal_index}, Elec {electrode_index}")
        ax.set_xlabel("X-dim")
        ax.set_ylabel("Y-dim")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Gráfico guardado en {save_path}")
        else:
            plt.show()

    def _plot_multiple_2d(self,
                          transform_type: str,
                          signal_index: int,
                          electrode_indices: list,
                          data_3d: np.ndarray,
                          save_path: Optional[str]):
        """
        Grafica varios mapas 2D submuestreados: data_3d => (C, dim1, dim2).
        """
        num_to_plot = len(electrode_indices)
        num_cols = 3
        num_rows = (num_to_plot + num_cols - 1) // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))
        axs = axs.flatten()

        for i, e_idx in enumerate(electrode_indices):
            arr_2d = data_3d[e_idx]
            im = axs[i].imshow(arr_2d, aspect='auto', origin='lower', cmap='viridis')
            axs[i].set_title(f"Elec {e_idx}")
            fig.colorbar(im, ax=axs[i], shrink=0.8)
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
        fig.suptitle(f"{transform_type.upper()} - Sig {signal_index}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Gráfico guardado en {save_path}")
        else:
            plt.show()

    def _plot_multiple_2d_alt(self,
                              transform_type: str,
                              signal_index: int,
                              electrode_indices: list,
                              data_3d: np.ndarray,
                              save_path: Optional[str]):
        """
        Versión análoga cuando data_3d => (dim1, dim2, C).
        """
        num_to_plot = len(electrode_indices)
        num_cols = 3
        num_rows = (num_to_plot + num_cols - 1)//num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows*3))
        axs = axs.flatten()

        for i, e_idx in enumerate(electrode_indices):
            arr_2d = data_3d[:, :, e_idx]
            im = axs[i].imshow(arr_2d, aspect='auto', origin='lower', cmap='viridis')
            axs[i].set_title(f"Elec {e_idx}")
            fig.colorbar(im, ax=axs[i], shrink=0.8)
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
        fig.suptitle(f"{transform_type.upper()} - Sig {signal_index}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Gráfico guardado en {save_path}")
        else:
            plt.show()

    def visualize_transform_subplots(
        self,
        transform_type: str,
        data_4d: Union[np.ndarray, torch.Tensor],
        signal_index: int = 0,
        num_to_plot: int = 6,
        num_cols: int = 3,
        save_path: Optional[str] = None
    ) -> None:
        """
        Genera subplots para un batch de la transformada 2D, 
        mostrando num_to_plot canales/electrodos.
        data_4d => (B, C, H, W) o (B, H, W, C).
        """
        if isinstance(data_4d, torch.Tensor):
            data_4d = data_4d.cpu().numpy()

        B = data_4d.shape[0]
        if not (0 <= signal_index < B):
            print(f"signal_index fuera de rango [0..{B-1}].")
            return

        # Heurística: dimension de canales en la segunda o última
        is_channel_in_dim1 = (data_4d.shape[1] < 300)

        if is_channel_in_dim1:
            # => (B, C, H, W)
            C = data_4d.shape[1]
            H = data_4d.shape[2]
            W = data_4d.shape[3]
            data_3d = data_4d[signal_index]  # (C, H, W)
        else:
            # => (B, H, W, C)
            C = data_4d.shape[3]
            H = data_4d.shape[1]
            W = data_4d.shape[2]
            data_3d = np.moveaxis(data_4d[signal_index], -1, 0)  # => (C, H, W)

        if num_to_plot > C:
            num_to_plot = C
        step = max(1, C // num_to_plot)
        selected_ch = list(range(0, C, step))[:num_to_plot]

        num_rows = (num_to_plot + num_cols - 1)//num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
        axs = axs.flatten()

        for i, ch_idx in enumerate(selected_ch):
            arr_2d = data_3d[ch_idx]  # (H, W)
            im = axs[i].imshow(arr_2d, aspect='auto', origin='lower', cmap='viridis')
            axs[i].set_title(f"{transform_type.upper()} - Batch {signal_index}, Ch {ch_idx}")
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f"{transform_type.upper()} - Signal {signal_index}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Gráfico guardado en: {save_path}")
        else:
            plt.show()
