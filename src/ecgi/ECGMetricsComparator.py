import torch
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Optional

class ECGMetricsComparator:
    """
    Clase para el cálculo de métricas en el dominio temporal y en dominios transformados,
    redimensionando internamente (batch, time, electrodes) a (batch*time, electrodes).
    Todas las métricas se devuelven como (media, desviación estándar) calculadas sobre los electrodos.
    """
    def __init__(self, fs: int = 1000):
        self.real_time: Optional[torch.Tensor] = None
        self.gen_time: Optional[torch.Tensor] = None
        self.real_cwt: Optional[torch.Tensor] = None
        self.gen_cwt: Optional[torch.Tensor] = None
        self.real_dwt: Optional[torch.Tensor] = None
        self.gen_dwt: Optional[torch.Tensor] = None
        self.fs = fs

    # Métodos para definir las señales de cada dominio
    def set_time_domain_signals(self, real_3d: torch.Tensor, gen_3d: torch.Tensor) -> None:
        """Establece las señales en dominio del tiempo (asegurando que tengan la misma forma)."""
        assert real_3d.shape == gen_3d.shape, "Las formas de las señales deben coincidir."
        self.real_time = real_3d
        self.gen_time = gen_3d

    def set_cwt_domain_signals(self, real_4d: torch.Tensor, gen_4d: torch.Tensor) -> None:
        """Establece las señales transformadas por CWT."""
        assert real_4d.shape == gen_4d.shape, "Las formas deben coincidir."
        self.real_cwt = real_4d
        self.gen_cwt = gen_4d

    def set_dwt_domain_signals(self, real_4d: torch.Tensor, gen_4d: torch.Tensor) -> None:
        """Establece las señales transformadas por DWT."""
        assert real_4d.shape == gen_4d.shape, "Las formas deben coincidir."
        self.real_dwt = real_4d
        self.gen_dwt = gen_4d

    def _reshape_signals(self, real: torch.Tensor, gen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Redimensiona (batch, time, electrodes) a (batch*time, electrodes)
        usando .reshape() en lugar de .view() para evitar problemas con la memoria no contigua.
        """
        return real.reshape(-1, real.size(-1)), gen.reshape(-1, gen.size(-1))

    # Métricas en dominio temporal
    def compute_time_mae(self) -> Tuple[float, float]:
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        mae = torch.mean(torch.abs(r - g), dim=0)
        return mae.mean().item(), mae.std().item()

    def compute_time_cc(self) -> Tuple[float, float]:
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        r_centered = r - torch.mean(r, dim=0, keepdim=True)
        g_centered = g - torch.mean(g, dim=0, keepdim=True)
        covariance = torch.mean(r_centered * g_centered, dim=0)
        std_r = torch.std(r, dim=0, unbiased=False)
        std_g = torch.std(g, dim=0, unbiased=False)
        cc = covariance / (std_r * std_g + 1e-12)
        return cc.mean().item(), cc.std().item()

    def compute_time_mse_rmse(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        mse = torch.mean((r - g) ** 2, dim=0)
        rmse = torch.sqrt(mse)
        return (mse.mean().item(), mse.std().item()), (rmse.mean().item(), rmse.std().item())

    def compute_time_snr(self) -> Tuple[float, float]:
        # Reshape signals
        r, g = self._reshape_signals(self.real_time, self.gen_time)
        
        # Compute signal power
        signal_power = torch.mean(r ** 2, dim=0)
        
        # Compute noise power with improved numerical stability
        noise_power = torch.mean((r - g) ** 2, dim=0)
        
        # Add numerical stability safeguards
        # 1. Ensure noise power is not too small
        # 2. Prevent division by zero
        # 3. Clip values to prevent extreme log values
        eps = 1e-8  # A smaller epsilon for more precise numerical stability
        
        # Compute SNR with clipping and stability
        safe_noise_power = torch.clamp(noise_power, min=eps)
        safe_signal_power = torch.clamp(signal_power, min=eps)
        
        # Compute SNR using log10
        snr = 10 * torch.log10(safe_signal_power / safe_noise_power)
        
        # Clip SNR to prevent extreme values
        snr = torch.clamp(snr, min=-100, max=100)
        return snr.mean().item(), snr.std().item()

    def compute_time_spectral_entropy_diff(self) -> Tuple[float, float]:
        def spectral_entropy(signal: np.ndarray) -> float:
            f, Pxx = welch(signal, fs=self.fs, nperseg=256)
            return entropy(Pxx / Pxx.sum())
        r = self.real_time.reshape(-1, self.real_time.size(-1)).cpu().numpy()
        g = self.gen_time.reshape(-1, self.gen_time.size(-1)).cpu().numpy()
        diffs = [abs(spectral_entropy(r[:, e]) - spectral_entropy(g[:, e])) for e in range(r.shape[1])]
        return np.mean(diffs), np.std(diffs)

    # Métricas para CWT
    def compute_cwt_2d_mse(self) -> Tuple[float, float]:
        r = self.real_cwt.reshape(-1, self.real_cwt.size(-1))
        g = self.gen_cwt.reshape(-1, self.gen_cwt.size(-1))
        mse = torch.mean((r - g) ** 2, dim=0)
        return mse.mean().item(), mse.std().item()

    def compute_cwt_2d_ssim(self) -> Tuple[float, float]:
        r = self.real_cwt.permute(0, 2, 1, 3).contiguous().reshape(-1, self.real_cwt.size(1), self.real_cwt.size(3))
        g = self.gen_cwt.permute(0, 2, 1, 3).contiguous().reshape(-1, self.gen_cwt.size(1), self.gen_cwt.size(3))
        r_np = r.cpu().numpy()
        g_np = g.cpu().numpy()
        ssim_values = []
        for e in range(r_np.shape[-1]):
            for i in range(r_np.shape[0]):
                ssim_val = ssim(
                    r_np[i, :, e],
                    g_np[i, :, e],
                    data_range=r_np[i, :, e].max() - r_np[i, :, e].min()
                )
                ssim_values.append(ssim_val)
        return np.mean(ssim_values), np.std(ssim_values)

    # Métricas para DWT
    def compute_dwt_energy_diff(self) -> Tuple[float, float]:
        r = self.real_dwt.reshape(-1, self.real_dwt.size(-1))
        g = self.gen_dwt.reshape(-1, self.gen_dwt.size(-1))
        en_diff = torch.abs(torch.sum(r, dim=0) - torch.sum(g, dim=0))
        return en_diff.mean().item(), en_diff.std().item()
