import logging
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    """
    Aplana todas las dimensiones salvo la temporal.
    Entrada: [N, C, T] | [C, T] | [T]
    Salida:  [M, T]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)           # [1, T]
    elif x.dim() == 2:
        pass                         # [C, T]
    elif x.dim() >= 3:
        x = x.flatten(0, -2)         # [N*C*(...), T]
    else:
        raise ValueError(f"Tensor con dims no soportadas: {x.shape}")
    return x


class ECGMetricsCalculator:
    """
    Métricas para señales ECG/EMG.
    Las funciones aceptan tensores PyTorch y calculan valores agregados.
    """

    @staticmethod
    def compute_mae(targets: torch.Tensor, preds: torch.Tensor) -> Tuple[float, float]:
        """Mean Absolute Error (MAE) y su desviación estándar sobre todos los elementos."""
        if targets.shape != preds.shape:
            raise ValueError(f"Shapes distintas: {targets.shape} vs {preds.shape}")
        err = torch.abs(targets - preds).reshape(-1)
        return err.mean().item(), err.std(unbiased=False).item()

    @staticmethod
    def compute_mse(targets: torch.Tensor, preds: torch.Tensor) -> Tuple[float, float]:
        """Mean Squared Error (MSE) y su desviación estándar sobre todos los elementos."""
        if targets.shape != preds.shape:
            raise ValueError(f"Shapes distintas: {targets.shape} vs {preds.shape}")
        se = (targets - preds).pow(2).reshape(-1)
        return se.mean().item(), se.std(unbiased=False).item()

    @staticmethod
    def compute_rmse(targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Root Mean Squared Error (RMSE) agregado."""
        mse, _ = ECGMetricsCalculator.compute_mse(targets, preds)
        return math.sqrt(mse)

    @staticmethod
    def compute_cc(targets: torch.Tensor, preds: torch.Tensor) -> Tuple[float, float]:
        """
        Coeficiente de correlación (tipo coseno/correlación de Pearson sin centrar por lote),
        calculado por traza (a lo largo de T) y promediado.
        Devuelve (mean, std) sobre todas las trazas M=prod(dims salvo T).
        """
        if targets.shape != preds.shape:
            raise ValueError(f"Shapes distintas: {targets.shape} vs {preds.shape}")

        x = _flatten_time(targets)  # [M, T]
        y = _flatten_time(preds)    # [M, T]
        eps = 1e-12

        # Centrado por traza
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)

        num = (x * y).sum(dim=1)                       # [M]
        den = (x.norm(dim=1) * y.norm(dim=1) + eps)    # [M]
        cc  = num / den                                 # [M]

        return cc.mean().item(), cc.std(unbiased=True).item() if cc.numel() > 1 else (cc.mean().item(), 0.0)

    @staticmethod
    def _spectral_cosine_similarity(a: Union[torch.Tensor, np.ndarray],
                                    b: Union[torch.Tensor, np.ndarray],
                                    return_std: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Coseno espectral entre |RFFT(a)| y |RFFT(b)|, vectorizado por traza.
        Entrada: [..., T]; se opera a lo largo de T y se promedia sobre el resto.
        """
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Entradas deben ser torch.Tensor o np.ndarray")
        if a.shape != b.shape:
            raise ValueError(f"Shapes distintas: {a.shape} vs {b.shape}")

        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)

        A = torch.fft.rfft(a, dim=-1).abs()
        B = torch.fft.rfft(b, dim=-1).abs()
        A = _flatten_time(A)  # [M, F]
        B = _flatten_time(B)  # [M, F]

        cos = F.cosine_similarity(A, B, dim=1, eps=1e-12)  # [M]
        mean = cos.mean().item()
        if return_std:
            std = cos.std(unbiased=True).item() if cos.numel() > 1 else 0.0
            return mean, std
        return mean

    @staticmethod
    def _max_cross_correlation(a: Union[torch.Tensor, np.ndarray],
                               b: Union[torch.Tensor, np.ndarray],
                               return_std: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Máxima correlación cruzada normalizada por traza (full lags), con FFT.
        Normalización: max(|r_xy|) / (T * std(a) * std(b) + eps)
        Entrada: [..., T]; se promedia sobre el resto de dims.
        """
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        if a.shape != b.shape:
            raise ValueError(f"Shapes distintas: {a.shape} vs {b.shape}")

        x = _flatten_time(a.to(dtype=torch.float32))  # [M, T]
        y = _flatten_time(b.to(dtype=torch.float32))  # [M, T]
        M, T = x.shape
        eps = 1e-12

        # Longitud para 'full' (2*T-1) usando padding a potencia de 2 para eficiencia
        full_len = 2 * T - 1
        L = 1 << (full_len - 1).bit_length()  # next pow2 ≥ full_len

        X = torch.fft.rfft(x, n=L, dim=1)                 # [M, Lr]
        Y = torch.fft.rfft(y, n=L, dim=1)
        corr_lin = torch.fft.irfft(X * torch.conj(Y), n=L, dim=1).real  # [M, L]

        # Reorganizar para ‘full’: [0..T-1] y [L-(T-1)..L-1]
        corr_full = torch.cat([corr_lin[:, :T], corr_lin[:, - (T - 1):]], dim=1)  # [M, 2T-1]

        # Normalización por energía
        denom = (T * x.std(dim=1, unbiased=False) * y.std(dim=1, unbiased=False) + eps)  # [M]
        mcc = (corr_full.abs().amax(dim=1) / denom)  # [M]

        mean = mcc.mean().item()
        if return_std:
            std = mcc.std(unbiased=True).item() if mcc.numel() > 1 else 0.0
            return mean, std
        return mean

    @staticmethod
    def complete_report(targets: torch.Tensor, preds: torch.Tensor) -> dict:
        """
        Reporte completo y consistente con logging:
        - Valores escalares o (mean, std) según la métrica.
        """
        if targets.shape != preds.shape:
            raise ValueError(f"Shapes distintas en reporte: {targets.shape} vs {preds.shape}")

        # Asegurar tensores CPU/float32 para operaciones FFT y reducción
        if targets.is_cuda:
            targets = targets.detach().cpu()
        else:
            targets = targets.detach()
        preds = preds.detach().cpu().to(dtype=targets.dtype)

        mae_mean, mae_std = ECGMetricsCalculator.compute_mae(targets, preds)
        mse_mean, mse_std = ECGMetricsCalculator.compute_mse(targets, preds)
        rmse = ECGMetricsCalculator.compute_rmse(targets, preds)
        cc_mean, cc_std = ECGMetricsCalculator.compute_cc(targets, preds)
        spec_mean, spec_std = ECGMetricsCalculator._spectral_cosine_similarity(
            targets, preds, return_std=True
        )
        xcorr_mean, xcorr_std = ECGMetricsCalculator._max_cross_correlation(
            targets, preds, return_std=True
        )

        return {
            "MAE": (mae_mean, mae_std),
            "MSE": (mse_mean, mse_std),
            "RMSE": rmse,
            "PCC": (cc_mean, cc_std),                 # nombre más explícito
            "SpectralCosine": (spec_mean, spec_std),
            "MaxCrossCorrelation": (xcorr_mean, xcorr_std),
        }
