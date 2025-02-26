import torch
import numpy as np
import logging
from scipy.signal import butter, filtfilt, resample
import pywt
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from typing import Optional, Union

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# CLASE SignalProcessorTorch (procesamiento en GPU)
###############################################################################
class SignalProcessorTorch:
    def __init__(
        self,
        data: torch.Tensor,
        fs: int,
        scales: torch.Tensor,
        wavelet: str = 'morl'
    ):
        """
        Inicializa el procesador de señales en GPU.

        Args:
            data: Tensor 3D (n_signals, n_samples, n_electrodes) preferiblemente en GPU.
            fs: Frecuencia de muestreo.
            scales: Tensor 1D con las escalas (por ejemplo, torch.arange(1, 128, device='cuda')).
            wavelet: Nombre de wavelet a usar en la CWT.
        """
        if data.ndim != 3:
            raise ValueError("El tensor 'data' debe ser 3D: (n_signals, n_samples, n_electrodes).")
        self.data = data
        self.fs = fs
        self.scales = scales
        self.wavelet = wavelet
        self.filtered_data: Optional[torch.Tensor] = None

    def apply_butter_filter(
        self,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        order: int = 4,
        btype: str = 'band'
    ) -> torch.Tensor:
        """
        Aplica un filtro Butterworth (utilizando filtfilt) en CPU y retorna un tensor en el dispositivo original.

        Args:
            lowcut: Frecuencia de corte inferior (para filtros 'band' o 'bandstop').
            highcut: Frecuencia de corte superior (para filtros 'band' o 'bandstop'; para 'low', se usa highcut).
            order: Orden del filtro.
            btype: Tipo de filtro ('band', 'bandstop', 'low' o 'high').

        Returns:
            Tensor filtrado en el mismo dispositivo que self.data.
        """
        self._validate_filter_parameters(lowcut, highcut, btype)
        data_cpu = self.data.detach().cpu().numpy()
        nyquist = 0.5 * self.fs
        b, a = self._design_filter(lowcut, highcut, nyquist, order, btype)
        # Aplicar el filtro a lo largo del eje de muestras (axis=1)
        filtered_cpu = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=data_cpu)
        filtered_tensor = torch.from_numpy(filtered_cpu).float().to(self.data.device)
        self.filtered_data = filtered_tensor
        logger.info("Filtro Butterworth aplicado (CPU -> GPU).")
        return filtered_tensor

    def _validate_filter_parameters(self, lowcut, highcut, btype):
        if btype in ['band', 'bandstop'] and (lowcut is None or highcut is None):
            raise ValueError("Faltan límites lowcut/highcut para filtro banda/bandstop.")
        if btype == 'low' and highcut is None:
            raise ValueError("Falta 'highcut' en filtro paso bajo.")
        if btype == 'high' and lowcut is None:
            raise ValueError("Falta 'lowcut' en filtro paso alto.")

    def _design_filter(self, lowcut, highcut, nyquist, order, btype):
        if btype in ['low', 'high']:
            cutoff = (highcut if btype == 'low' else lowcut) / nyquist
            return butter(order, cutoff, btype=btype)
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype=btype)

    def compute_wavelet_tensor(
        self,
        transform_type: str = 'cwt',
        wavelet: str = 'db4',
        level: Optional[int] = None,
        target_length: Optional[int] = None,
        use_filtered: bool = False
    ) -> torch.Tensor:
        """
        Realiza la transformada (CWT o DWT) en CPU usando pywt y retorna un tensor en el dispositivo original.

        Args:
            transform_type: 'cwt' o 'dwt'.
            wavelet: Wavelet a usar para DWT.
            level: Nivel de descomposición para DWT.
            target_length: Longitud de salida deseada.
            use_filtered: Si es True, usa self.filtered_data; de lo contrario, self.data.

        Returns:
            Tensor resultante de la transformada en el dispositivo de self.data.
        """
        source_tensor = self.filtered_data if (use_filtered and self.filtered_data is not None) else self.data
        source_cpu = source_tensor.detach().cpu().numpy()
        if transform_type == 'cwt':
            out_np = self._compute_cwt_cpu(source_cpu, target_length)
        elif transform_type == 'dwt':
            out_np = self._compute_dwt_cpu(source_cpu, wavelet, level, target_length)
        else:
            raise ValueError(f"transform_type no soportado: {transform_type}")
        out_tensor = torch.from_numpy(out_np).float().to(self.data.device)
        return out_tensor

    def _compute_cwt_cpu(self, data_cpu: np.ndarray, target_length: Optional[int]):
        n_signals, n_samples, n_electrodes = data_cpu.shape
        scales_np = self.scales.detach().cpu().numpy()
        n_scales = len(scales_np)
        target_length = target_length or n_samples
        cwt_out = np.zeros((n_signals, n_scales, target_length, n_electrodes), dtype=np.float32)
        for s in range(n_signals):
            for e in range(n_electrodes):
                coefs, _ = pywt.cwt(
                    data_cpu[s, :, e],
                    scales_np,
                    self.wavelet,
                    sampling_period=1/self.fs
                )
                mag = np.abs(coefs)
                cwt_out[s, :, :, e] = self._resample_if_needed(mag, target_length)
        return cwt_out

    def _compute_dwt_cpu(self, data_cpu: np.ndarray, wavelet: str, level: Optional[int], target_length: Optional[int]):
        n_signals, n_samples, n_electrodes = data_cpu.shape
        if level is None:
            wave_obj = pywt.Wavelet(wavelet)
            level = pywt.dwt_max_level(n_samples, wave_obj.dec_len)
        target_length = target_length or n_samples
        dwt_out = np.zeros((n_signals, level, target_length, n_electrodes), dtype=np.float32)
        for s in range(n_signals):
            for e in range(n_electrodes):
                coeffs = pywt.wavedec(data_cpu[s, :, e], wavelet, level=level)
                details = coeffs[1:]
                detail_stack = []
                for d in details:
                    d_res = resample(d, target_length) if d.shape[0] != target_length else d
                    detail_stack.append(d_res)
                dwt_out[s, :, :, e] = np.stack(detail_stack, axis=0)
        return dwt_out

    def _resample_if_needed(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        if arr.shape[1] != target_len:
            return resample(arr, target_len, axis=-1)
        return arr

    def reduce_electrodes_kpca(
        self,
        var_threshold: float = 0.95,
        kernel: str = 'rbf',
        gamma: float = 0.01,
        degree: int = 3,
        coef0: float = 1.0
    ) -> np.ndarray:
        """
        Aplica Kernel PCA sobre la dimensión de electrodos utilizando scikit-learn (en CPU)
        y retorna un array con forma (n_signals, n_samples, k) donde k es el número de componentes
        necesarias para alcanzar var_threshold de varianza.
        """
        n_signals, n_samples, n_electrodes = self.data.shape
        N = n_signals * n_samples
        # Aplanar para obtener una matriz 2D (N, n_electrodes)
        X_2D = np.asarray(self.data.cpu(), dtype=np.float64).reshape(N, n_electrodes)
        kpca = KernelPCA(
            n_components=n_electrodes,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            fit_inverse_transform=True,
            eigen_solver='dense'
        )
        X_kpca = kpca.fit_transform(X_2D)
        try:
            lambdas = kpca.lambdas_
        except AttributeError:
            logger.warning("'lambdas_' no disponible en KernelPCA; usando varianza de X_kpca.")
            lambdas = np.var(X_kpca, axis=0)
        total_var = np.sum(lambdas)
        cumulative = np.cumsum(lambdas) / total_var
        k = np.searchsorted(cumulative, var_threshold) + 1
        X_kpca_reduced = X_kpca[:, :k]
        data_kpca = X_kpca_reduced.reshape(n_signals, n_samples, k)
        logger.info(f"Kernel PCA: Retención={cumulative[k-1]*100:.2f}%, k={k} de {n_electrodes}, kernel={kernel}.")
        return data_kpca