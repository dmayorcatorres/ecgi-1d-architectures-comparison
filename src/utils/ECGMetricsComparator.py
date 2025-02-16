import torch
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Optional

class ECGMetricsComparator:
    """
    Clase para cálculo de métricas agrupando batch y tiempo:
      - Internamente redimensiona (batch, time, electrodes) -> (batch*time, electrodes)
      - Todas las métricas devuelven (mean, std) sobre los electrodos
    """

    def __init__(self, fs=1000):
        self.real_time: Optional[torch.Tensor] = None
        self.gen_time: Optional[torch.Tensor] = None
        self.real_cwt: Optional[torch.Tensor] = None
        self.gen_cwt: Optional[torch.Tensor] = None
        self.real_dwt: Optional[torch.Tensor] = None
        self.gen_dwt: Optional[torch.Tensor] = None
        self.fs = fs

    # ---------------------------------------------------------------------
    # 1) MÉTODOS PARA ALMACENAR SEÑALES
    # ---------------------------------------------------------------------
    def set_time_domain_signals(self, real_3d: torch.Tensor, gen_3d: torch.Tensor):
        assert real_3d.shape == gen_3d.shape, "Shapes must match"
        self.real_time = real_3d
        self.gen_time = gen_3d

    def set_cwt_domain_signals(self, real_4d: torch.Tensor, gen_4d: torch.Tensor):
        assert real_4d.shape == gen_4d.shape, "Shapes must match"
        self.real_cwt = real_4d
        self.gen_cwt = gen_4d

    def set_dwt_domain_signals(self, real_4d: torch.Tensor, gen_4d: torch.Tensor):
        assert real_4d.shape == gen_4d.shape, "Shapes must match"
        self.real_dwt = real_4d
        self.gen_dwt = gen_4d

    def _reshape_signals(self, real: torch.Tensor, gen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Redimensiona (batch, time, electrodes) -> (batch*time, electrodes)"""
        return (
            real.view(-1, real.size(-1)),
            gen.view(-1, gen.size(-1))
        )

    # ---------------------------------------------------------------------
    # 2) MÉTRICAS EN DOMINIO TEMPORAL
    # ---------------------------------------------------------------------
    def compute_time_mae(self) -> Tuple[float, float]:
        """MAE (media y std sobre electrodos)"""
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        mae = torch.mean(torch.abs(r - g), dim=0)
        return mae.mean().item(), mae.std().item()

    def compute_time_cc(self) -> Tuple[float, float]:
        """Correlación de Pearson (media y std)"""
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        r_centered = r - torch.mean(r, dim=0, keepdim=True)
        g_centered = g - torch.mean(g, dim=0, keepdim=True)
        covariance = torch.mean(r_centered * g_centered, dim=0)
        std_r = torch.std(r, dim=0, unbiased=False)
        std_g = torch.std(g, dim=0, unbiased=False)
        cc = covariance / (std_r * std_g + 1e-12)
        return cc.mean().item(), cc.std().item()

    def compute_time_mse_rmse(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """MSE y RMSE (media y std para ambas)"""
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        mse = torch.mean((r - g)**2, dim=0)
        rmse = torch.sqrt(mse)
        return (
            (mse.mean().item(), mse.std().item()),
            (rmse.mean().item(), rmse.std().item())
        )

    def compute_time_snr(self) -> Tuple[float, float]:
        """SNR en dB (media y std)"""
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        signal_power = torch.mean(r**2, dim=0)
        noise_power = torch.mean((r - g)**2, dim=0)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-12))
        return snr.mean().item(), snr.std().item()

    def compute_time_spectral_entropy_diff(self) -> Tuple[float, float]:
        """Diferencia de entropía espectral (media y std)"""
        def spectral_entropy(signal: np.ndarray):
            f, Pxx = welch(signal, fs=self.fs, nperseg=256)
            return entropy(Pxx/Pxx.sum())

        r = self.real_time.view(-1, self.real_time.size(-1)).numpy()
        g = self.gen_time.view(-1, self.gen_time.size(-1)).numpy()

        diffs = []
        for e in range(r.shape[1]):
            diffs.append(abs(
                spectral_entropy(r[:, e]) - spectral_entropy(g[:, e])
            ))
        return np.mean(diffs), np.std(diffs)

    # ---------------------------------------------------------------------
    # 3) MÉTRICAS PARA CWT
    # ---------------------------------------------------------------------
    def compute_cwt_2d_mse(self) -> Tuple[float, float]:
        """MSE 2D en escalogramas (media y std)"""
        r = self.real_cwt.view(-1, self.real_cwt.size(-1))
        g = self.gen_cwt.view(-1, self.gen_cwt.size(-1))
        mse = torch.mean((r - g)**2, dim=0)
        return mse.mean().item(), mse.std().item()

    def compute_cwt_2d_ssim(self) -> Tuple[float, float]:
        """SSIM 2D en escalogramas (media y std)"""
        r = self.real_cwt.permute(0, 2, 1, 3).contiguous().view(-1, self.real_cwt.size(1), self.real_cwt.size(3))
        g = self.gen_cwt.permute(0, 2, 1, 3).contiguous().view(-1, self.gen_cwt.size(1), self.gen_cwt.size(3))

        r_np = r.numpy()
        g_np = g.numpy()

        ssim_values = []
        for e in range(r_np.shape[-1]):
            for i in range(r_np.shape[0]):
                ssim_values.append(ssim(
                    r_np[i, :, e],
                    g_np[i, :, e],
                    data_range=r_np[i, :, e].max()-r_np[i, :, e].min()
                ))
        return np.mean(ssim_values), np.std(ssim_values)

    # ---------------------------------------------------------------------
    # 4) MÉTRICAS PARA DWT
    # ---------------------------------------------------------------------
    def compute_dwt_energy_diff(self) -> Tuple[float, float]:
        """Diferencia de energía en coeficientes DWT (media y std)"""
        r = self.real_dwt.view(-1, self.real_dwt.size(-1))
        g = self.gen_dwt.view(-1, self.gen_dwt.size(-1))
        en_diff = torch.abs(torch.sum(r, dim=0) - torch.sum(g, dim=0))
        return en_diff.mean().item(), en_diff.std().item()
