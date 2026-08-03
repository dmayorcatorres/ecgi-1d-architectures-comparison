import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Series Decomposition
# -------------------------
class SeriesDecomposition(nn.Module):
    """
    Media móvil con padding 'reflect' adaptativo para evitar sesgo en bordes.
    Opera canal-a-canal sobre (B, D, T).
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size debe ser impar"
        self.k = kernel_size

    def forward(self, x: torch.Tensor):
        # x: (B, D, T)
        T = x.shape[-1]
        pad = min(self.k // 2, max(T - 1, 0))
        if pad > 0:
            x_pad = F.pad(x, (pad, pad), mode="reflect")
        else:
            x_pad = x
        # kernel efectivo (si T es muy pequeño)
        k_eff = 2 * pad + 1
        trend = F.avg_pool1d(x_pad, kernel_size=k_eff, stride=1)  # (B, D, T)
        seasonal = x - trend
        return seasonal, trend


# --------------------------------------
# 2) Auto-Correlation (versión simplificada)
#    FFT + Top-k time-delay aggregation
# --------------------------------------
class AutoCorrelationLayer(nn.Module):
    """
    Aproximación del mecanismo de Autoformer:
      - Proyecciones Q, K, V (multi-head)
      - Correlación temporal vía FFT (q_fft * conj(k_fft))
      - Top-k retardos globales por cabeza
      - Agregación de V con "roll" temporal y pesos softmax
    """
    def __init__(self, d_model: int, n_heads: int, top_k: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _global_topk_lags(self, corr: torch.Tensor, k: int):
        """
        corr: (B, H, T) → se promedia en batch y se eligen k retardos por cabeza.
        Devuelve idx: (H, k) en [0, T-1]
        """
        corr_mean = corr.mean(dim=0)                 # (H, T)
        _, idx = torch.topk(corr_mean, k=k, dim=-1)  # (H, k)
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape
        k_eff = min(self.top_k, T)

        # Proyecciones y reshape a (B, H, T, dk)
        Q = self.q_proj(x).view(B, T, self.h, self.dk).permute(0, 2, 1, 3)  # (B,H,T,dk)
        K = self.k_proj(x).view(B, T, self.h, self.dk).permute(0, 2, 1, 3)  # (B,H,T,dk)
        V = self.v_proj(x).view(B, T, self.h, self.dk).permute(0, 2, 1, 3)  # (B,H,T,dk)

        # FFT sobre eje temporal → correlación circular por retardos
        # torch.fft.rfft/irfft no soportan bfloat16/float16 (RuntimeError:
        # "Unsupported dtype BFloat16"), y utils/runtime.py::amp_config()
        # activa bf16 automáticamente en GPUs que lo soportan. Se fuerza
        # float32 solo para el bloque de FFT y se vuelve al dtype original
        # (el de V) al terminar, para no romper el resto del forward bajo
        # autocast (ver MIGRATION_REPORT.md, hallazgo "autoformer1d").
        orig_dtype = Q.dtype
        Qf = torch.fft.rfft(Q.permute(0, 1, 3, 2).float(), dim=-1)  # (B,H,dk, T//2+1)
        Kf = torch.fft.rfft(K.permute(0, 1, 3, 2).float(), dim=-1)  # (B,H,dk, T//2+1)

        # Escalado por sqrt(dk) para estabilidad
        scale = math.sqrt(self.dk)
        Corr = torch.fft.irfft((Qf * torch.conj(Kf)) / scale, n=T, dim=-1).sum(dim=2)  # (B,H,T)
        Corr = Corr.to(orig_dtype)

        # Top-k retardos globales por cabeza
        idx = self._global_topk_lags(Corr, k_eff)  # (H, k_eff)

        # Pesos softmax (globales) en el dtype de V
        with torch.no_grad():
            corr_mean = Corr.mean(dim=0)  # (H, T)
            w = torch.stack([corr_mean[h, idx[h]] for h in range(self.h)], dim=0)  # (H,k_eff)
            w = torch.softmax(w, dim=-1).to(V.dtype)

        # Agregación por time-delay
        out = torch.zeros_like(V)
        for h in range(self.h):
            acc = torch.zeros_like(V[:, h])  # (B,T,dk)
            for j in range(k_eff):
                lag = int(idx[h, j].item())
                v_shift = torch.roll(V[:, h], shifts=lag, dims=1)  # (B,T,dk)
                acc = acc + w[h, j].view(1, 1, 1) * v_shift
            out[:, h] = acc

        # Volver a (B,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.o_proj(self.dropout(out))
        return out


# -------------------------
# 3) Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    """
    Bloque estilo Autoformer:
      - Decomposition → seasonal, trend
      - AutoCorrelation sobre seasonal + residual + norm
      - FFN (GELU) + residual + norm
      - Decomposition de salida (progresiva)
    Devuelve: seasonal_out, trend_out (ambos (B,T,D))
    """
    def __init__(self, d_model: int, n_heads: int, kernel_size: int = 25,
                 dropout: float = 0.1, top_k: int = 4, ffn_ratio: int = 4):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.decomp2 = SeriesDecomposition(kernel_size)

        self.auto_corr = AutoCorrelationLayer(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        returns: seasonal, trend  (ambos (B,T,D))
        """
        # 1) Decomposition inicial (trabajando en (B,D,T))
        seasonal, trend1 = self.decomp1(x.transpose(1, 2))  # (B,D,T)
        seasonal = seasonal.transpose(1, 2)                 # (B,T,D)

        # 2) Auto-Correlation sobre seasonal
        z = self.auto_corr(seasonal)
        z = self.dropout(z)
        seasonal = self.norm1(seasonal + z)

        # 3) FFN + residual
        z = self.ffn(seasonal)
        z = self.dropout(z)
        seasonal = self.norm2(seasonal + z)

        # 4) Decomposition progresiva y suma de tendencias con ejes consistentes
        seasonal, trend2 = self.decomp2(seasonal.transpose(1, 2))  # (B,D,T)
        seasonal = seasonal.transpose(1, 2)                        # (B,T,D)
        trend = (trend1 + trend2).transpose(1, 2)                  # -> (B,T,D)

        return seasonal, trend


# -------------------------
# 4) Autoformer1D (Encoder)
# -------------------------
class Autoformer1D(nn.Module):
    """
    Autoformer1D (solo encoder, salida same-length):
      in : (B, C_in, T)
      out: (B, C_out, T)

    Flujo:
      Conv1x1 → (B,D,T)
      Permuta a (B,T,D)
      N×[Decomp + AutoCorr + FFN + Decomp] acumulando trend
      Reconstruye y + trend acumulado
      Conv1x1 final → (B,C_out,T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.1,
        top_k: int = 4,
        ffn_ratio: int = 4,
        **kwargs
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        assert kernel_size % 2 == 1, "kernel_size debe ser impar"

        self.time_steps = int(time_steps)
        self.in_proj  = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, kernel_size, dropout, top_k=top_k, ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        B, C, T = x.shape
        assert T == self.time_steps, f"Se esperaba T={self.time_steps}, llegó {T}"

        y = self.in_proj(x)                   # (B, D, T)
        y = y.permute(0, 2, 1).contiguous()   # (B, T, D)

        trend_acc = torch.zeros_like(y)       # (B, T, D)
        for layer in self.layers:
            y, trend = layer(y)               # (B,T,D), (B,T,D)
            trend_acc = trend_acc + trend

        # Reconstrucción: seasonal + tendencia acumulada
        y = (y + trend_acc).permute(0, 2, 1).contiguous()  # (B, D, T)
        return self.out_proj(y)                            # (B, C_out, T)
