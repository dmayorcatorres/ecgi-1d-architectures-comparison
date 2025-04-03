import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Union, List
import logging

from src.ecgi.TimeSeriesTo2DTransform import TimeSeriesTo2DTransform

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class ECGiVisualizer:
    """
    Visualiza series ECG/EMG y transformaciones 2D (RPM, GAF, CWT, etc.), 
    además de funciones para trazar mallas 3D.
    """

    def __init__(self, fs: int = 1000):
        """
        Args:
            fs: Frecuencia de muestreo por defecto para transformaciones.
        """
        self.transformer = TimeSeriesTo2DTransform(fs=fs)

    @staticmethod
    def plot_electrodes(
        original_data_3d: Union[np.ndarray, torch.Tensor],
        fs: int = 1000,
        signal_label: str = 'ECG',
        signal_index: int = 0,
        electrode_index: Optional[int] = None,
        num_to_plot: Optional[int] = None,
        reconstructed_data_3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
        save_path: Optional[str] = None  # Si se especifica, se guarda la figura
    ) -> None:
        # Convertir tensores a NumPy si es necesario.
        if isinstance(original_data_3d, torch.Tensor):
            original_data_3d = original_data_3d.cpu().numpy()
        if isinstance(reconstructed_data_3d, torch.Tensor):
            reconstructed_data_3d = reconstructed_data_3d.cpu().numpy()

        # Nuevo orden: (n_signals, n_electrodes, n_time)
        n_signals, n_electrodes, n_time = original_data_3d.shape
        if not (0 <= signal_index < n_signals):
            logger.error(f"signal_index out of range: 0..{n_signals-1}")
            return

        has_reconstruction = False
        if reconstructed_data_3d is not None:
            rec_ns, rec_ne, rec_nt = reconstructed_data_3d.shape
            if rec_ns == n_signals and rec_ne == n_electrodes and rec_nt == n_time:
                has_reconstruction = True
            else:
                logger.warning("Shape mismatch in 'reconstructed_data_3d'. Reconstruction ignored.")
                reconstructed_data_3d = None

        # Modo 1: Visualización de un solo electrodo.
        if electrode_index is not None:
            if not (0 <= electrode_index < n_electrodes):
                logger.error(f"electrode_index out of range: 0..{n_electrodes-1}")
                return

            # Nuevo orden: acceder [signal_index, electrode_index, :]
            original_series = original_data_3d[signal_index, electrode_index, :]
            time_axis = np.arange(n_time) / fs

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, original_series, color='blue', label='Original')

            if has_reconstruction:
                rec_series = reconstructed_data_3d[signal_index, electrode_index, :]
                ax.plot(time_axis, rec_series, color='red', label='Reconstructed')

            ax.set_title(f'{signal_label} - Signal {signal_index}, Electrode {electrode_index}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True)
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        # Modo 2: Submuestreo de electrodos.
        if num_to_plot is not None:
            if num_to_plot <= 0:
                logger.error("num_to_plot must be > 0.")
                return

            # Ahora cada señal es de forma (n_electrodes, n_time)
            signal_2d = original_data_3d[signal_index]
            step = max(1, n_electrodes // num_to_plot)
            selected_electrodes = list(range(0, n_electrodes, step))[:num_to_plot]

            time_axis = np.arange(n_time) / fs
            num_cols = 3
            num_rows = (num_to_plot + num_cols - 1) // num_cols

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))
            axs = axs.flatten()
            for i, e_idx in enumerate(selected_electrodes):
                axs[i].plot(time_axis, signal_2d[e_idx, :], color='blue', label='Original')
                if has_reconstruction:
                    rec_2d = reconstructed_data_3d[signal_index]
                    axs[i].plot(time_axis, rec_2d[e_idx, :], color='red', label='Reconstructed')
                axs[i].set_title(f'Electrode {e_idx}')
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel('Amplitude')
                axs[i].legend()
                axs[i].grid(True)
            # Ocultar ejes vacíos si existen
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            fig.suptitle(f'{signal_label} - Signal {signal_index} (Subsampling {num_to_plot} electrodes)',
                         fontsize=16, y=1.02)
            fig.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Either 'electrode_index' (single electrode) or 'num_to_plot' (subsampling) must be specified.")


    @staticmethod
    def plot_structure_mesh(
        nodes: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        structure_label: str = "Structure"
    ) -> None:
        try:
            nodes = np.array(nodes)
            faces = np.array(faces, dtype=int)
        except Exception as e:
            logger.error(f"Error processing nodes or faces: {e}")
            return

        if nodes.ndim == 2 and nodes.shape[0] == 3 and nodes.shape[1] > 3:
            nodes = nodes.T

        if faces.ndim == 2 and faces.shape[0] == 3 and faces.shape[1] > 3:
            faces = faces.T

        if np.min(faces) == 1:
            faces = faces - 1

        # Filtrar caras inválidas
        polys = []
        for face in faces:
            if len(face) != 3:
                logger.warning(f"Se omite cara inválida (no tiene 3 índices): {face}")
                continue
            if np.any(face < 0) or np.any(face >= nodes.shape[0]):
                logger.warning(f"Se omite cara con índices fuera de rango: {face}")
                continue
            polys.append(nodes[face])

        if len(polys) == 0:
            logger.error("No se encontraron caras válidas para la visualización.")
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(polys, alpha=0.5, edgecolor='k', linewidths=0.5)
        mesh.set_facecolor('cyan')
        ax.add_collection3d(mesh)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='red', s=10)
        margin = 0.1
        ax.set_xlim(np.min(nodes[:, 0]) - margin, np.max(nodes[:, 0]) + margin)
        ax.set_ylim(np.min(nodes[:, 1]) - margin, np.max(nodes[:, 1]) + margin)
        ax.set_zlim(np.min(nodes[:, 2]) - margin, np.max(nodes[:, 2]) + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{structure_label} Mesh")
        plt.tight_layout()
        plt.show()
