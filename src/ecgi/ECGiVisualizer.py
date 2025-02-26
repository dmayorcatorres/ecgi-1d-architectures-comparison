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

    # =====================================================
    #   Métodos GET -> devuelven el tensor transformado
    # =====================================================

    def get_rpm(self, data_3d: Union[np.ndarray, torch.Tensor],
                paa_factor: int = 1) -> torch.Tensor:
        """
        Aplica la transformación 'rpm' y retorna el tensor 4D: (B, C, dim1, dim2).
        data_3d => (n_signals, n_electrodes, n_time)
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        rpm_out = self.transformer.transform(
            x=data_3d,
            method='rpm',
            paa_factor=paa_factor
        )
        return rpm_out

    def get_gaf(self, data_3d: Union[np.ndarray, torch.Tensor],
                mode: str = 'summation',  # <--- Cambiar el nombre del parámetro
                paa_factor: int = 1) -> torch.Tensor:
        """
        Aplica GAF (Gramian Angular Field). 'mode' puede ser 'summation' o 'difference'
        Retorna tensor => (B, C, time, time).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()

        gaf_out = self.transformer.transform(
            x=data_3d,
            method='gaf',        # Indica que queremos la transformación 'gaf'
            paa_factor=paa_factor,
            gaf_mode=mode        # <-- Pasar un kwarg que no choque con 'method'
        )
        return gaf_out


    def get_mtf(self, data_3d, n_bins=8, paa_factor=1) -> torch.Tensor:
        """
        Retorna MTF transform => (B, C, time, time).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        mtf_out = self.transformer.transform(
            x=data_3d,
            method='mtf',
            paa_factor=paa_factor,
            n_bins=n_bins
        )
        return mtf_out
    
    def get_rp(self, data_3d: Union[np.ndarray, torch.Tensor],
            threshold: float = 0.2,
            paa_factor: int = 1,
            metric: str = 'euclidean') -> torch.Tensor:
        """
        Recurrence Plot => (B, C, time, time).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        return self.transformer.transform(
            data_3d,
            method='rp',
            paa_factor=paa_factor,
            threshold=threshold,
            metric=metric
        )
    
    def get_jrp(self, data_3d: Union[np.ndarray, torch.Tensor],
            embedding_dim: int = 3,
            time_delay: int = 1,
            threshold: float = 0.1,
            paa_factor: int = 1) -> torch.Tensor:
        """
        Joint Recurrence Plot => (B, C, emb_len, emb_len).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        return self.transformer.transform(
            data_3d,
            method='jrp',
            paa_factor=paa_factor,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            threshold=threshold
        )

    def get_cwt(self, data_3d, wavelet_type='morlet',
                paa_factor=1, **kwargs) -> torch.Tensor:
        """
        Retorna CWT => (B, n_scales, time, channels).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        cwt_out = self.transformer.transform(
            x=data_3d,
            method='cwt',
            paa_factor=paa_factor,
            wavelet_type=wavelet_type,
            **kwargs
        )
        return cwt_out

    def get_dwt(self, data_3d, wavelet='db4', paa_factor=1, **kwargs) -> torch.Tensor:
        """
        Retorna DWT => (B, n_levels, time, channels).
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        dwt_out = self.transformer.transform(
            x=data_3d,
            method='dwt',
            paa_factor=paa_factor,
            wavelet=wavelet,
            **kwargs
        )
        return dwt_out

    def get_spectrogram(self,
                        data_3d: Union[np.ndarray, torch.Tensor],
                        nperseg: int = 32,
                        noverlap: Optional[int] = None,
                        paa_factor: int = 1) -> torch.Tensor:
        """
        Aplica la transformación 'spectrogram' => (B, freq_bins, time_bins, channels).
        data_3d => (n_signals, n_electrodes, n_time)
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        spectro_out = self.transformer.transform(
            x=data_3d,
            method='spectrogram',
            paa_factor=paa_factor,
            nperseg=nperseg,
            noverlap=noverlap
        )
        return spectro_out


    def get_stft(self,
                data_3d: Union[np.ndarray, torch.Tensor],
                window_type: str = 'hann',
                nperseg: int = 256,
                noverlap: Optional[int] = None,
                paa_factor: int = 1) -> torch.Tensor:
        """
        Aplica 'stft' => (B, freq_bins, time_bins, channels).
        data_3d => (n_signals, n_electrodes, n_time)
        """
        if isinstance(data_3d, np.ndarray):
            data_3d = torch.from_numpy(data_3d).float()
        stft_out = self.transformer.transform(
            x=data_3d,
            method='stft',
            paa_factor=paa_factor,
            window_type=window_type,
            nperseg=nperseg,
            noverlap=noverlap
        )
        return stft_out

    def visualize_spectrogram(self,
                            spectro_tensor: torch.Tensor,
                            signal_index: int = 0,
                            electrode_index: Optional[int] = None,
                            num_to_plot: Optional[int] = None,
                            title_label: str = 'Spectrogram',
                            save_path: Optional[str] = None) -> None:
        """
        Visualiza el espectrograma => (B, freq_bins, time_bins, channels).
        """
        B, freq_bins, time_bins, C = spectro_tensor.shape
        if not (0 <= signal_index < B):
            logger.error("signal_index out of range.")
            return

        # Modo 1: un electrodo
        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                logger.error("electrode_index out of range.")
                return
            spec_2d = spectro_tensor[signal_index, :, :, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(spec_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time bins")
            ax.set_ylabel("Frequency bins")
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        # Modo 2: varios electrodos
        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i, e_idx in enumerate(selected):
                spec_2d = spectro_tensor[signal_index, :, :, e_idx].detach().cpu().numpy()
                im=axs[i].imshow(spec_2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Electrode {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for spectrogram visualization.")


    def visualize_stft(self,
                    stft_tensor: torch.Tensor,
                    signal_index: int = 0,
                    electrode_index: Optional[int] = None,
                    num_to_plot: Optional[int] = None,
                    title_label: str = 'STFT',
                    save_path: Optional[str] = None) -> None:
        """
        Visualiza la STFT => (B, freq_bins, time_bins, channels).
        Semejante a 'visualize_spectrogram'.
        """
        B, freq_bins, time_bins, C = stft_tensor.shape
        if not (0 <= signal_index < B):
            logger.error("signal_index out of range.")
            return

        # Modo 1: un electrodo
        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                logger.error("electrode_index out of range.")
                return
            stft_2d = stft_tensor[signal_index, :, :, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im= ax.imshow(stft_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time bins")
            ax.set_ylabel("Frequency bins")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        # Modo 2: varios electrodos
        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i,e_idx in enumerate(selected):
                stft_2d = stft_tensor[signal_index, :, :, e_idx].detach().cpu().numpy()
                im=axs[i].imshow(stft_2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Elec {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for STFT visualization.")


    def visualize_rpm(self, 
                      rpm_tensor: torch.Tensor,
                      signal_index: int = 0,
                      electrode_index: Optional[int] = None,
                      num_to_plot: Optional[int] = None,
                      title_label: str = 'RPM',
                      save_path: Optional[str] = None) -> None:
        
        B, C, dim1, dim2 = rpm_tensor.shape
        if not (0 <= signal_index < B):
            print("signal_index out of range.")
            return
        
        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                print("electrode_index out of range.")
                return
            rpm_2d = rpm_tensor[signal_index, electrode_index, :, :].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(rpm_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Signal {signal_index}, Electrode {electrode_index}")
            ax.set_xlabel("RPM dimension 1")
            ax.set_ylabel("RPM dimension 2")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            
        elif num_to_plot is not None and num_to_plot > 0:
            step = max(1, C // num_to_plot)
            selected = list(range(0, C, step))[:num_to_plot]
            num_cols = 3
            num_rows = (len(selected) + num_cols - 1) // num_cols
            
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))
            axs = axs.flatten()
            for i, e_idx in enumerate(selected):
                rpm_2d = rpm_tensor[signal_index, e_idx, :, :].cpu().numpy()
                im = axs[i].imshow(rpm_2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Electrode {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1, len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Signal {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        
        else:
            print("Specify either electrode_index or num_to_plot.")
            
    def visualize_gaf(self,
                      gaf_tensor: torch.Tensor,
                      signal_index: int = 0,
                      electrode_index: Optional[int] = None,
                      num_to_plot: Optional[int] = None,
                      title_label: str = 'GAF',
                      save_path: Optional[str] = None) -> None:
        """
        gaf_tensor => (B, C, T, T)
        Parecido a RPM => (B, C, dim1, dim2)
        """
        B, C, dim1, dim2 = gaf_tensor.shape
        if not (0 <= signal_index < B):
            logger.error("signal_index out of range.")
            return

        # Modo 1: un electrodo
        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                logger.error("electrode_index out of range.")
                return
            gaf_2d = gaf_tensor[signal_index, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(gaf_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time dimension")
            ax.set_ylabel("Time dimension")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        # Modo 2: varios electrodos
        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i,e_idx in enumerate(selected):
                arr2d = gaf_tensor[signal_index,e_idx].detach().cpu().numpy()
                im=axs[i].imshow(arr2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Elec {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for GAF visualization.")

    def visualize_mtf(self,
                      mtf_tensor: torch.Tensor,
                      signal_index: int = 0,
                      electrode_index: Optional[int] = None,
                      num_to_plot: Optional[int] = None,
                      title_label: str = 'MTF',
                      save_path: Optional[str] = None) -> None:
        """
        mtf_tensor => (B, C, time, time)
        Visualización similar a GAF, RPM
        """
        B, C, dim1, dim2 = mtf_tensor.shape
        if not (0<=signal_index<B):
            logger.error("signal_index out of range.")
            return
        if electrode_index is not None:
            if not (0<=electrode_index<C):
                logger.error("electrode_index out of range.")
                return
            mtf_2d = mtf_tensor[signal_index, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im= ax.imshow(mtf_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time dimension")
            ax.set_ylabel("Time dimension")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i,e_idx in enumerate(selected):
                arr2d = mtf_tensor[signal_index,e_idx].detach().cpu().numpy()
                im=axs[i].imshow(arr2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Elec {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for MTF visualization.")

    def visualize_rp(self,
                     rp_tensor: torch.Tensor,
                     signal_index: int = 0,
                     electrode_index: Optional[int] = None,
                     num_to_plot: Optional[int] = None,
                     title_label: str = 'RP',
                     save_path: Optional[str] = None) -> None:
        """
        rp_tensor => (B, C, time, time)
        """
        B, C, dim1, dim2 = rp_tensor.shape
        if not (0<=signal_index<B):
            logger.error("signal_index out of range.")
            return

        # Modo 1: un electrodo
        if electrode_index is not None:
            if not (0<=electrode_index<C):
                logger.error("electrode_index out of range.")
                return
            rp_2d = rp_tensor[signal_index, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im= ax.imshow(rp_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time dimension")
            ax.set_ylabel("Time dimension")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        # Modo 2: varios
        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i,e_idx in enumerate(selected):
                arr2d = rp_tensor[signal_index,e_idx].detach().cpu().numpy()
                im=axs[i].imshow(arr2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Elec {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for RP visualization.")

    def visualize_jrp(self,
                      jrp_tensor: torch.Tensor,
                      signal_index: int = 0,
                      electrode_index: Optional[int] = None,
                      num_to_plot: Optional[int] = None,
                      title_label: str = 'JRP',
                      save_path: Optional[str] = None) -> None:
        """
        jrp_tensor => (B, C, emb_len, emb_len)
        """
        B, C, dim1, dim2 = jrp_tensor.shape
        if not (0<=signal_index<B):
            logger.error("signal_index out of range.")
            return

        # Modo 1: un electrodo
        if electrode_index is not None:
            if not (0<=electrode_index<C):
                logger.error("electrode_index out of range.")
                return
            jrp_2d = jrp_tensor[signal_index, electrode_index].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8,6))
            im= ax.imshow(jrp_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Sig {signal_index}, Elec {electrode_index}")
            ax.set_xlabel("Time dimension(embedding)")
            ax.set_ylabel("Time dimension(embedding)")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            return

        # Modo 2: varios
        if num_to_plot is not None and num_to_plot>0:
            step = max(1, C//num_to_plot)
            selected = list(range(0,C,step))[:num_to_plot]
            num_cols=3
            num_rows=(len(selected)+num_cols-1)//num_cols
            fig, axs=plt.subplots(num_rows,num_cols, figsize=(14,num_rows*3))
            axs=axs.flatten()
            for i,e_idx in enumerate(selected):
                arr2d = jrp_tensor[signal_index,e_idx].detach().cpu().numpy()
                im=axs[i].imshow(arr2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Elec {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1,len(axs)):
                axs[j].axis('off')
            fig.suptitle(f"{title_label} - Sig {signal_index}", fontsize=16)
            plt.tight_layout()
           
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return

        logger.error("Specify either electrode_index or num_to_plot for JRP visualization.")

    def visualize_cwt(self,
                      cwt_tensor: torch.Tensor,
                      signal_index: int = 0,  
                      electrode_index: Optional[int] = None,
                      num_to_plot: Optional[int] = None,
                      title_label: str = 'CWT',
                      save_path: Optional[str] = None) -> None:
        
        B, dim1, dim2, C = cwt_tensor.shape
        if not (0 <= signal_index < B):
            print("signal_index out of range.") 
            return
        
        if electrode_index is not None:
            if not (0 <= electrode_index < C):
                print("electrode_index out of range.")
                return
            cwt_2d = cwt_tensor[signal_index, :, :, electrode_index].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cwt_2d, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title_label} - Signal {signal_index}, Electrode {electrode_index}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Scales")
            
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            
        elif num_to_plot is not None and num_to_plot > 0:
            step = max(1, C // num_to_plot)
            selected = list(range(0, C, step))[:num_to_plot]
            num_cols = 3
            num_rows = (len(selected) + num_cols - 1) // num_cols
            
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))
            axs = axs.flatten()
            for i, e_idx in enumerate(selected):
                cwt_2d = cwt_tensor[signal_index, :, :, e_idx].cpu().numpy()
                im = axs[i].imshow(cwt_2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Electrode {e_idx}')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i+1, len(axs)):
                axs[j].axis('off')    
            fig.suptitle(f"{title_label} - Signal {signal_index}", fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        
        else:
            print("Specify either electrode_index or num_to_plot.")


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
    def visualize_dwt(
        dwt_tensor_4d: Union[np.ndarray, torch.Tensor],
        signal_index: int = 0,
        electrode_index: Optional[int] = None,
        num_to_plot: Optional[int] = None,
        wavelet_label: str = 'DWT',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plots the Discrete Wavelet Transform (DWT) in 2D (levels x time) for one or multiple electrodes.
        Se asume que la forma del tensor es (n_signals, n_levels, canal, time).
        """
        if isinstance(dwt_tensor_4d, torch.Tensor):
            dwt_tensor_4d = dwt_tensor_4d.cpu().numpy()

        # Nuevo orden: (n_signals, n_levels, n_electrodes, n_time)
        n_signals, n_levels, n_electrodes, n_time = dwt_tensor_4d.shape
        if not (0 <= signal_index < n_signals):
            logger.error("Signal index out of range.")
            return

        # Modo de un solo electrodo.
        if electrode_index is not None:
            if not (0 <= electrode_index < n_electrodes):
                logger.error("Electrode index out of range.")
                return
            # Acceder [signal_index, :, electrode_index, :]
            dwt_2d = dwt_tensor_4d[signal_index, :, electrode_index, :]
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(dwt_2d, aspect='auto', cmap='viridis', origin='lower')
            fig.colorbar(im, ax=ax, label='Amplitude')
            ax.set_title(f'{wavelet_label} - Signal {signal_index}, Electrode {electrode_index}')
            ax.set_xlabel('Time axis')
            ax.set_ylabel('Levels')
            if save_path:
                fig.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"DWT plot saved to {save_path}")
            else:
                plt.show()
        
        # Modo de múltiples electrodos (subplots)
        elif num_to_plot is not None:
            if num_to_plot <= 0:
                logger.error("num_to_plot must be greater than 0.")
                return
            step = max(1, n_electrodes // num_to_plot)
            selected_electrodes = list(range(0, n_electrodes, step))[:num_to_plot]
            num_cols = 3
            num_rows = (len(selected_electrodes) + num_cols - 1) // num_cols

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 3))
            axs = axs.flatten()
            for i, e_idx in enumerate(selected_electrodes):
                # Acceder [signal_index, :, e_idx, :]
                dwt_2d = dwt_tensor_4d[signal_index, :, e_idx, :]
                im = axs[i].imshow(dwt_2d, aspect='auto', cmap='viridis', origin='lower')
                axs[i].set_title(f'Electrode {e_idx}')
                axs[i].set_xlabel('Time axis')
                axs[i].set_ylabel('Levels')
                fig.colorbar(im, ax=axs[i], shrink=0.8)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
            fig.suptitle(f'{wavelet_label} - Signal {signal_index} (Subsampling {len(selected_electrodes)} electrodes)', fontsize=16, y=1.02)
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"DWT multi-electrode plot saved to {save_path}")
            else:
                plt.show()
        
        else:
            logger.error("Either 'electrode_index' or 'num_to_plot' must be specified for DWT plotting.")

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
