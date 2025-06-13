"""
ECGMetricsComparator
────────────────────
Implementación libre de efectos colaterales: cada *compute_* calcula y
devuelve, pero **no** sobre-escribe el propio método ni ningún atributo.
De ese modo, se puede reutilizar la misma instancia en varias llamadas
sin provocar el error «'int' object is not callable».
"""
from __future__ import annotations
from typing import Tuple, Optional

import torch
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim


class ECGMetricsComparator:
    def __init__(self, fs: int = 1000) -> None:
        self.fs = fs
        # señales (batch, time, electrodes)   ó  (batch, scales, time, electrodes)
        self.real_time: Optional[torch.Tensor] = None
        self.gen_time:  Optional[torch.Tensor] = None
        self.real_cwt:  Optional[torch.Tensor] = None
        self.gen_cwt:   Optional[torch.Tensor] = None
        self.real_dwt:  Optional[torch.Tensor] = None
        self.gen_dwt:   Optional[torch.Tensor] = None

    # ───────────────────── setters ─────────────────────
    def set_time_domain_signals(self,
                                real_3d: torch.Tensor,
                                gen_3d:  torch.Tensor) -> None:
        assert real_3d.shape == gen_3d.shape, "Las formas deben coincidir"
        self.real_time, self.gen_time = real_3d, gen_3d

    def set_cwt_domain_signals(self,
                               real_4d: torch.Tensor,
                               gen_4d:  torch.Tensor) -> None:
        assert real_4d.shape == gen_4d.shape, "Las formas deben coincidir"
        self.real_cwt, self.gen_cwt = real_4d, gen_4d

    def set_dwt_domain_signals(self,
                               real_4d: torch.Tensor,
                               gen_4d:  torch.Tensor) -> None:
        assert real_4d.shape == gen_4d.shape, "Las formas deben coincidir"
        self.real_dwt, self.gen_dwt = real_4d, gen_4d

    # ───────────────────── helpers ─────────────────────
    @staticmethod
    def _reshape_bt_electrodes(x: torch.Tensor) -> torch.Tensor:
        # (batch, time, electrodes) -> (batch*time, electrodes)
        return x.reshape(-1, x.size(-1))

    # ───────── métricas - dominio temporal ─────────
    def compute_time_mae(self) -> Tuple[float, float]:
        r = self._reshape_bt_electrodes(self.real_time)
        g = self._reshape_bt_electrodes(self.gen_time)
        mae = torch.mean(torch.abs(r - g), dim=0)
        return mae.mean().item(), mae.std().item()

    def compute_time_cc(self) -> Tuple[float, float]:
        r = self._reshape_bt_electrodes(self.real_time)
        g = self._reshape_bt_electrodes(self.gen_time)
        r_c, g_c = r - r.mean(0, keepdim=True), g - g.mean(0, keepdim=True)
        cc = (r_c * g_c).mean(0) / (r.std(0) * g.std(0) + 1e-12)
        return cc.mean().item(), cc.std().item()

    def compute_time_mse_rmse(self) -> Tuple[Tuple[float, float],
                                             Tuple[float, float]]:
        r = self._reshape_bt_electrodes(self.real_time)
        g = self._reshape_bt_electrodes(self.gen_time)
        mse = torch.mean((r - g) ** 2, dim=0)
        rmse = torch.sqrt(mse)
        return (mse.mean().item(),  mse.std().item()), \
               (rmse.mean().item(), rmse.std().item())

    def compute_time_snr(self) -> Tuple[float, float]:
        r = self._reshape_bt_electrodes(self.real_time)
        g = self._reshape_bt_electrodes(self.gen_time)
        sig_pow  = torch.mean(r ** 2, dim=0)
        noise_pow = torch.mean((r - g) ** 2, dim=0).clamp_min(1e-8)
        snr = 10 * torch.log10(sig_pow.clamp_min(1e-8) / noise_pow)
        snr = torch.clamp(snr, -100, 100)
        return snr.mean().item(), snr.std().item()

    def compute_time_spectral_entropy_diff(self) -> Tuple[float, float]:
        def spec_entropy(x: np.ndarray) -> float:
            f, pxx = welch(x, fs=self.fs, nperseg=256)
            return entropy(pxx / pxx.sum())

        r = self._reshape_bt_electrodes(self.real_time).cpu().numpy()
        g = self._reshape_bt_electrodes(self.gen_time).cpu().numpy()
        diffs = [abs(spec_entropy(r[:, e]) - spec_entropy(g[:, e]))
                 for e in range(r.shape[1])]
        return np.mean(diffs), np.std(diffs)

    # ───────── métricas - CWT ─────────
    def compute_cwt_2d_mse(self) -> Tuple[float, float]:
        r = self.real_cwt.reshape(-1, self.real_cwt.size(-1))
        g = self.gen_cwt.reshape(-1, self.gen_cwt.size(-1))
        mse = torch.mean((r - g) ** 2, dim=0)
        return mse.mean().item(), mse.std().item()

    def compute_cwt_2d_ssim(self) -> Tuple[float, float]:
        r = self.real_cwt.permute(0, 2, 1, 3).reshape(-1,
                                                      self.real_cwt.size(1),
                                                      self.real_cwt.size(3))
        g = self.gen_cwt.permute(0, 2, 1, 3).reshape(-1,
                                                     self.gen_cwt.size(1),
                                                     self.gen_cwt.size(3))
        r_np, g_np = r.cpu().numpy(), g.cpu().numpy()
        vals = [ssim(r_np[i], g_np[i], data_range=r_np[i].ptp())
                for i in range(r_np.shape[0])]
        return float(np.mean(vals)), float(np.std(vals))

    # ───────── métricas - DWT ─────────
    def compute_dwt_energy_diff(self) -> Tuple[float, float]:
        r = self.real_dwt.reshape(-1, self.real_dwt.size(-1))
        g = self.gen_dwt.reshape(-1, self.gen_dwt.size(-1))
        diff = torch.abs(r.sum(0) - g.sum(0))
        return diff.mean().item(), diff.std().item()
