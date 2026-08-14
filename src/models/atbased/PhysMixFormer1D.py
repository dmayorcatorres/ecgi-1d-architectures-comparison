# models/atbased/PhysMixFormer1D.py
#
# Portado de aux/physformer_standalone/physmixformer_standalone_v3.py
# (backbone Transformer1D/MixFormer1D multiescala + gauge + Π
# fila-estocástica + rama física de difusión Chebyshev con warm-up +
# pérdida compuesta de 5 términos ponderada por incertidumbre). Validado
# manualmente contra los .mat reales de auckland/bordeaux (PCC ~0.95 /
# ~0.86) antes de esta integración -- ver historial de la sesión que
# depuró el standalone.
#
# Reemplaza a la variante 37 "physformer" (antes models/atbased/
# PhysFormer.py, que no incluía física por no tener el pipeline acceso a
# geometría de malla). Este archivo SÍ la usa cuando está disponible
# (data/geometry.py::load_geometry) y se degrada automáticamente a un
# backbone puramente data-driven cuando no (ver attach_geometry).
#
# Diferencia de contrato con el resto de las 36 arquitecturas: forward(x)
# devuelve SIEMPRE una tupla (x_hat, Pi, A_eff), no un tensor plano, y
# expone compute_loss()/uncertainty_parameters()/anneal_epoch() para el
# camino de entrenamiento especial gateado por USES_COMPOSITE_LOSS en
# run_bohb.py (train_holdout/final_train).
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.geometry import GeometryPack, gauge_projector_matrix, gauge_weights_from_degree


def apply_gauge(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """x: (B,E,T) -> P·x sobre el eje de electrodos."""
    return torch.einsum('ij,bjt->bit', P, x)


# ═══════════════════════ Backbone: Transformer1D / MixFormer1D ═══════════════

class PositionalEncodingSin(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, :x.size(-1)]


class RotaryPE(nn.Module):
    def __init__(self, head_dim: int, base_theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requiere head_dim par"
        self.head_dim = head_dim
        self.base_theta = base_theta
        self.register_buffer("_cached_t", torch.tensor(0), persistent=False)
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def _build_cache(self, T: int, device, dtype):
        half = self.head_dim // 2
        inv_freq = (self.base_theta ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum('t,f->tf', t, inv_freq)
        cos = torch.cos(freqs)[None, None, :, :]
        sin = torch.sin(freqs)[None, None, :, :]
        self._cached_t = torch.tensor(T, device=device)
        self._cos = cos.to(dtype=dtype)
        self._sin = sin.to(dtype=dtype)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        device, dtype = x.device, x.dtype
        if (self._cos.numel() == 0) or (self._cached_t.item() < T):
            self._build_cache(T, device, dtype)
        cos = self._cos[:, :, :T, :]
        sin = self._sin[:, :, :T, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x2 * cos + x1 * sin
        return x_rot


class MSWrapper(nn.Module):
    def __init__(self, module: nn.Module, scales=(1, 2, 4)):
        super().__init__()
        self.module = module
        self.scales = scales
        self.alpha = nn.Parameter(torch.zeros(len(scales)))

    def _down(self, x_btD: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1: return x_btD
        return F.avg_pool1d(x_btD.transpose(1, 2), kernel_size=s, stride=s).transpose(1, 2)

    def _up(self, x_btD: torch.Tensor, T: int, s: int) -> torch.Tensor:
        if s == 1: return x_btD
        return F.interpolate(x_btD.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        outs = []
        for s in self.scales:
            xs = self._down(x, s)
            ys = self.module(xs)
            ys = self._up(ys, T, s)
            outs.append(ys)
        w = self.alpha.softmax(dim=0)
        return sum(w[i] * outs[i] for i in range(len(outs)))


class SelfAttention1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.1,
                 pe_type: str = "rope", rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.pe_type = pe_type.lower()
        self.rope = RotaryPE(self.head_dim, base_theta=rope_theta) if self.pe_type == "rope" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = to_heads(q); k = to_heads(k); v = to_heads(v)
        if self.rope is not None:
            q = self.rope.apply_rope(q)
            k = self.rope.apply_rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(ctx)


class InceptionMultiScale1D(nn.Module):
    """Kernels {7,15,31,41}/dilataciones {1,2,3,5}."""
    def __init__(self, d_model: int, kernels=(7, 15, 31, 41), dilations=(1, 2, 3, 5),
                 use_maxpool: bool = True, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, dilation=d, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU()
            ))
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.GELU()
        )
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.max_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.GELU()
            )
        n_branches = len(self.branches) + 1 + (1 if use_maxpool else 0)
        self.mix = nn.Conv1d(d_model * n_branches, d_model, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm1d(d_model) if norm == "bn" else (nn.GroupNorm(1, d_model) if norm == "gn" else nn.Identity())
        self.dropout = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, max(1, d_model // 4), 1), nn.GELU(),
            nn.Conv1d(max(1, d_model // 4), d_model, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        outs.append(self.pool_branch(x))
        if self.use_maxpool:
            outs.append(self.max_branch(x))
        y = torch.cat(outs, dim=1)
        y = self.mix(y)
        y = self.norm(y)
        gate = self.se(y)
        y = y * gate
        return self.dropout(y)


class ConvSandwich1D(nn.Module):
    def __init__(self, d_model: int, kernel: int = 7, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model) if norm == "bn" else (nn.GroupNorm(1, d_model) if norm == "gn" else nn.Identity())
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bdT: torch.Tensor) -> torch.Tensor:
        y = self.dw(x_bdT)
        y = self.pw(y)
        y = self.act(y)
        y = self.norm(y)
        return self.drop(y)


class TransformerBlock1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 pe_type: str = "rope", rope_theta: float = 10000.0,
                 attn_ms_scales=(1, 2, 4), conv_kernel: int = 7, conv_norm: str = "gn"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        base_attn = SelfAttention1D(d_model, num_heads, attn_drop=dropout, pe_type=pe_type, rope_theta=rope_theta)
        self.attn = MSWrapper(base_attn, scales=attn_ms_scales)
        self.conv_sandwich = ConvSandwich1D(d_model, kernel=conv_kernel, norm=conv_norm, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        xc = self.conv_sandwich(x.permute(0, 2, 1))
        x = x + xc.permute(0, 2, 1)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer1D(nn.Module):
    """f_θ(Y): (B,Et,T) -> (B,Ee,T). `use_ms_temporal=False` da el backbone
    "Transformer1D" puro (ablación); `True` da "MixFormer1D" (multiescala
    sin física)."""
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                 dropout: float = 0.1, pe_type: str = "rope", rope_theta: float = 10000.0,
                 use_ms_temporal: bool = True, ms_type: str = "inception",
                 ms_kernels=(7, 15, 31, 41), ms_dilations=(1, 2, 3, 5),
                 attn_ms_scales=(1, 2, 4)):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncodingSin(d_model, max_len=time_steps) if pe_type.lower() == "sin" else None
        self.use_ms_temporal = use_ms_temporal
        if use_ms_temporal:
            if ms_type == "inception":
                self.ms_temporal = InceptionMultiScale1D(d_model, kernels=ms_kernels, dilations=ms_dilations,
                                                         use_maxpool=True, norm="gn", dropout=dropout)
            else:
                raise ValueError(f"ms_type desconocido: {ms_type}")
        else:
            self.ms_temporal = None
        self.layers = nn.ModuleList([
            TransformerBlock1D(d_model, num_heads, dropout=dropout, pe_type=pe_type, rope_theta=rope_theta,
                               attn_ms_scales=attn_ms_scales, conv_kernel=7, conv_norm="gn")
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, y_torso: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(y_torso)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        if self.ms_temporal is not None:
            x = x + self.ms_temporal(x)
        x = x.permute(0, 2, 1)
        for blk in self.layers:
            x = blk(x)
        x = self.out_ln(x)
        x = x.permute(0, 2, 1).contiguous()
        return self.output_proj(x)


# ═══════════════════════ Π fila-estocástica + rama física ════════════════════

class RowStochasticCoupling(nn.Module):
    """Π ∈ R^{Et×Ee}: Π≥0, Π·1_Ee=1_Et por construcción (softmax por fila).
    SIEMPRE mapea epicardio→torso: y_torso ≈ Π · x_epi."""
    def __init__(self, D_te: torch.Tensor, p: float = 2.0,
                 tau0: float = 1.0, tau_end: float = 0.3):
        super().__init__()
        self.register_buffer("D_te", D_te)            # (Et, Ee)
        self.p = p
        self.tau0 = tau0
        self.tau_end = tau_end
        self.tau = tau0
        self.logits = nn.Parameter(torch.zeros_like(D_te))  # residuo aprendible

    def anneal_tau(self, epoch: int, total_epochs: int):
        frac = min(max(epoch / max(1, total_epochs), 0.0), 1.0)
        self.tau = self.tau0 * (self.tau_end / self.tau0) ** frac

    def forward(self) -> torch.Tensor:
        prior_logits = -(self.D_te.clamp_min(0).pow(self.p)) / max(self.tau, 1e-6)
        logits = prior_logits + self.logits
        Pi = F.softmax(logits, dim=-1)  # (Et, Ee), fila-estocástica garantizada
        return Pi


class PhysicsBranch(nn.Module):
    """A_eff = Σ_j γ_j · T_j(L_e)·Πᵀ;  X_phys = A_eff · Y.

    Warm-up progresivo: los gate logits {β_j} que determinan {γ_j} vía
    softmax rampan LINEALMENTE desde un valor ~uniforme hasta los logits
    aprendidos (`gamma_logits`), durante las primeras `warmup_epochs`
    épocas -- evita que A_eff entre a la fusión residual desde la época 1
    con una mezcla de bases sin calibrar."""
    def __init__(self, L_e: torch.Tensor, coupling: RowStochasticCoupling,
                 n_bases: int = 5, lmax: float = 2.0,
                 warmup_epochs: int = 8, warmup_uniform_logit: float = 0.1):
        super().__init__()
        Ee = L_e.shape[0]
        self.register_buffer("L_e", L_e)
        L_cheb = (2.0 / lmax) * L_e - torch.eye(Ee, device=L_e.device, dtype=L_e.dtype)
        self.register_buffer("L_cheb", L_cheb)
        self.coupling = coupling
        self.n_bases = max(1, n_bases)
        self.gamma_logits = nn.Parameter(torch.zeros(self.n_bases, device=L_e.device, dtype=L_e.dtype))
        self.warmup_epochs = max(1, warmup_epochs)
        self.warmup_uniform_logit = warmup_uniform_logit
        self.warmup_alpha = 1.0  # 1.0 = sin warm-up activo; se pisa cada época vía anneal_warmup()

    def anneal_warmup(self, epoch: int, total_epochs: Optional[int] = None):
        """1-indexado. alpha=0 en la época 1 (casi uniforme) -> alpha=1 al
        llegar a `warmup_epochs` (logits totalmente aprendidos)."""
        w = self.warmup_epochs if total_epochs is None else min(self.warmup_epochs, max(1, total_epochs))
        self.warmup_alpha = min(max((epoch - 1) / max(1, w - 1) if w > 1 else 1.0, 0.0), 1.0)

    def _effective_gamma_logits(self) -> torch.Tensor:
        if self.warmup_alpha >= 1.0:
            return self.gamma_logits
        uniform = torch.full_like(self.gamma_logits, self.warmup_uniform_logit)
        return (1.0 - self.warmup_alpha) * uniform + self.warmup_alpha * self.gamma_logits

    def _chebyshev_bases(self, Pi_T: torch.Tensor) -> List[torch.Tensor]:
        bases = [Pi_T]
        if self.n_bases > 1:
            Tk_prev, Tk = Pi_T, self.L_cheb @ Pi_T
            bases.append(Tk)
            for _ in range(2, self.n_bases):
                Tk_next = 2.0 * (self.L_cheb @ Tk) - Tk_prev
                bases.append(Tk_next)
                Tk_prev, Tk = Tk, Tk_next
        return bases

    def forward(self, y_torso_mV: torch.Tensor):
        """y_torso_mV: (B,Et,T) señal de torso en mV (gauge-proyectada, SIN
        z-score -- la mezcla de Π solo es físicamente significativa en
        unidades reales compartidas)."""
        Pi = self.coupling()                 # (Et, Ee)
        Pi_T = Pi.transpose(0, 1)             # (Ee, Et)
        bases = self._chebyshev_bases(Pi_T)   # cada uno (Ee, Et)
        gamma = F.softmax(self._effective_gamma_logits(), dim=0)
        A_eff = sum(g * B for g, B in zip(gamma, bases))       # (Ee, Et)
        X_phys = torch.einsum('ec,bct->bet', A_eff, y_torso_mV)  # (B,Ee,T)
        return X_phys, Pi, A_eff


# ═══════════════════════ Pérdida compuesta (5 términos) ═══════════════════════

class CompositeLossTerms(nn.Module):
    """time/spectral/corr/geom/smooth, SIN combinar -- devuelve
    {nombre: tensor_escalar con grad}. La combinación (ponderación por
    incertidumbre) vive en UncertaintyWeightedCombiner."""
    def __init__(self, fs: float = 250.0, band: Tuple[float, float] = (0.5, 100.0)):
        super().__init__()
        self.fs = fs
        self.band = band

    def time_term(self, x_hat, x, omega_e):
        w = omega_e.view(1, -1, 1)
        return ((x_hat - x) ** 2 * w).mean()

    def spectral_term(self, x_hat, x):
        # torch.fft.rfft no soporta bfloat16; se fuerza float32 solo para
        # el tramo de FFT (mismo patrón que Autoformer1D, commit de2b1d7d).
        Xh = torch.fft.rfft(x_hat.float(), dim=-1)
        Xt = torch.fft.rfft(x.float(), dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        band_mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if band_mask.sum() == 0:
            return x.new_tensor(0.0)
        diff = (Xh - Xt)[..., band_mask]
        return (diff.real ** 2 + diff.imag ** 2).mean()

    def corr_term(self, x_hat, x, eps: float = 1e-8):
        xh = x_hat - x_hat.mean(-1, keepdim=True)
        xt = x - x.mean(-1, keepdim=True)
        num = (xh * xt).sum(-1)
        den = torch.sqrt((xh ** 2).sum(-1) * (xt ** 2).sum(-1) + eps)
        rho = num / (den + eps)
        return (1.0 - rho).mean()

    def geom_term(self, x_hat, y_torso, Pi, omega_t):
        # Pi: (Et,Ee); x_hat: (B,Ee,T) -> y_pred: (B,Et,T)
        y_pred = torch.einsum('oe,bet->bot', Pi, x_hat)
        w = omega_t.view(1, -1, 1)
        return ((y_pred - y_torso) ** 2 * w).mean()

    def smooth_term(self, x_hat, L_e):
        # x^T L_e x por muestra y tiempo, NO ||Lx||^2
        xh = x_hat.permute(0, 2, 1)              # (B,T,Ee)
        Lx = torch.einsum('ij,btj->bti', L_e, xh)  # (B,T,Ee)
        val = (xh * Lx).sum(-1)                   # (B,T)
        return val.mean()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
               y_torso: Optional[torch.Tensor] = None, Pi: Optional[torch.Tensor] = None,
               L_e: Optional[torch.Tensor] = None,
               omega_e: Optional[torch.Tensor] = None,
               omega_t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if omega_e is None:
            omega_e = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
        terms: Dict[str, torch.Tensor] = {
            "time": self.time_term(x_hat, x, omega_e),
            "spec": self.spectral_term(x_hat, x),
            "corr": self.corr_term(x_hat, x),
        }
        if (Pi is not None) and (y_torso is not None):
            if omega_t is None:
                omega_t = torch.ones(y_torso.shape[1], device=x.device, dtype=x.dtype)
            terms["geom"] = self.geom_term(x_hat, y_torso, Pi, omega_t)
        if L_e is not None:
            terms["smooth"] = self.smooth_term(x_hat, L_e)
        return terms


class UncertaintyWeightedCombiner(nn.Module):
    """L_total = Σ_k exp(-s_k)·L_k + s_k   (Kendall, Gal & Cipolla, CVPR 2018).

    NOTA -- el clamp `s_k = self.log_vars[name].clamp(*self.s_clamp)` corta
    el gradiente por completo fuera de [-6,6] (torch.clamp da grad=0 ahí),
    así que en la práctica los log-vars quedan fijos en el primer batch
    donde su valor crudo se sale del rango, no "convergen" a él. Esto se
    dejó DELIBERADAMENTE sin corregir en esta integración: se probó una
    versión con proyección post-step (gradiente completo) + lr más bajo
    del optimizador de incertidumbre, y dio resultados PEORES en auckland
    y bordeaux (PCC 0.86/0.73 vs 0.95/0.86 con este comportamiento) -- ver
    historial de la sesión que lo depuró. Se mantiene tal cual v3 por ser
    lo empíricamente validado, no por ser la implementación "correcta" del
    método de Kendall et al."""
    def __init__(self, task_names: List[str], init_log_var: float = 0.0,
                s_clamp: Tuple[float, float] = (-6.0, 6.0)):
        super().__init__()
        self.s_clamp = s_clamp
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_var))) for name in task_names
        })  # init=0.0 -> peso inicial=1 para todos (arranque neutro)

    def forward(self, terms: Dict[str, torch.Tensor]):
        total = None
        log: Dict[str, Dict[str, float]] = {}
        for name, Lk in terms.items():
            if name not in self.log_vars:
                raise KeyError(f"'{name}' no está en UncertaintyWeightedCombiner.log_vars "
                               f"({list(self.log_vars.keys())}); pasar task_names completo al construirlo.")
            s_k = self.log_vars[name].clamp(*self.s_clamp)
            precision = torch.exp(-s_k)
            contrib = precision * Lk + s_k
            total = contrib if total is None else total + contrib
            log[name] = {"raw": Lk.item(), "weight": precision.item(), "s": s_k.item()}
        return total, log


# ═══════════════════════ Modelo completo ══════════════════════════════════════

class PhysMixFormer1D(nn.Module):
    """f_θ: torso (BSPM) -> corazón (HSPM), dirección fija.

    Contrato distinto al resto del pipeline: forward(x) devuelve SIEMPRE
    (x_hat, Pi, A_eff) -- Pi/A_eff son None sin geometría adjunta. La
    pérdida no es la MSELoss compartida: usar compute_loss(). Ver
    USES_COMPOSITE_LOSS y el camino especial en run_bohb.py
    (train_holdout/final_train).

    Sin attach_geometry() el modelo opera en modo puramente data-driven
    (Transformer1D/MixFormer1D sin física) -- así es como se
    auto-desactiva la física en datasets sin geometría de malla confiable
    (ver data/geometry.py::load_geometry, que devuelve None en ese caso).
    """
    USES_COMPOSITE_LOSS = True

    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 4,
                 dropout: float = 0.1, ablation: str = "physmixformer",
                 fuse_physics: bool = True, lambda_phys: float = 0.2,
                 lambda_phys_trainable: bool = False,
                 n_cheb_bases: int = 5, physics_warmup_epochs: int = 8,
                 **kwargs):
        super().__init__()
        use_ms = ablation in ("mixformer", "physmixformer")
        self.backbone = Transformer1D(
            time_steps=time_steps, input_channels=input_channels, output_channels=output_channels,
            d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout,
            pe_type="rope", use_ms_temporal=use_ms, ms_type="inception",
        )
        self.physics: Optional[PhysicsBranch] = None
        self._n_cheb_bases = n_cheb_bases
        self._physics_warmup_epochs = physics_warmup_epochs
        self._fuse_physics_requested = fuse_physics
        self.fuse_physics = False  # se activa en attach_geometry si hay física

        if lambda_phys_trainable:
            self.lambda_phys = nn.Parameter(torch.tensor(float(lambda_phys)))
        else:
            self.register_buffer("lambda_phys", torch.tensor(float(lambda_phys)))

        self.loss_terms = CompositeLossTerms(fs=250.0)
        self.combiner = UncertaintyWeightedCombiner(["time", "spec", "corr"])

    def attach_geometry(self, geometry: GeometryPack) -> None:
        """Llamar UNA vez, justo después de factory.create() (y de mover el
        modelo a DEVICE), antes de entrenar. No-op si nunca se llama."""
        device = next(self.backbone.parameters()).device

        D_te = geometry.D_te.to(device)
        L_heart = geometry.L_heart.to(device)
        coupling = RowStochasticCoupling(D_te)
        self.physics = PhysicsBranch(L_heart, coupling, n_bases=self._n_cheb_bases,
                                     warmup_epochs=self._physics_warmup_epochs)
        self.physics = self.physics.to(device)
        self.fuse_physics = self._fuse_physics_requested

        omega_t = gauge_weights_from_degree(geometry.deg_torso).to(device)
        omega_e = gauge_weights_from_degree(geometry.deg_heart).to(device)
        self.register_buffer("P_torso", gauge_projector_matrix(omega_t))
        self.register_buffer("P_epi", gauge_projector_matrix(omega_e))
        self.register_buffer("L_e", L_heart)
        self.register_buffer("omega_e", omega_e)
        self.register_buffer("omega_t", omega_t)

        # extender el combinador con los 2 términos que dependen de geometría
        self.combiner.log_vars["geom"] = nn.Parameter(torch.tensor(0.0, device=device))
        self.combiner.log_vars["smooth"] = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, x: torch.Tensor):
        """x: (B, input_channels, T) señal de torso cruda (sin normalizar,
        ver data/preprocessing.py -- el pipeline no hace z-score). Devuelve
        (x_hat, Pi, A_eff)."""
        eps = 1e-6
        x_z = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)

        y_in = x
        if hasattr(self, "P_torso"):
            y_in = apply_gauge(y_in, self.P_torso)

        x_base = self.backbone(x_z)

        Pi = A_eff = None
        if self.physics is not None:
            x_phys, Pi, A_eff = self.physics(y_in)
            if self.fuse_physics:
                lam = torch.clamp(self.lambda_phys, 0.0, 1.0)
                x_comb = x_base + lam * x_phys
            else:
                x_comb = x_base
        else:
            x_comb = x_base

        x_hat = x_comb
        if hasattr(self, "P_epi"):
            x_hat = apply_gauge(x_comb, self.P_epi)
        return x_hat, Pi, A_eff

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     x_input: torch.Tensor, Pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """pred/Pi: salida de forward(x_input). target: yb del batch."""
        T = pred.shape[-1]
        x_input = x_input[..., :T]
        target = target[..., :T]

        y_torso = x_input
        if hasattr(self, "P_torso"):
            y_torso = apply_gauge(x_input, self.P_torso)

        terms = self.loss_terms(
            pred, target,
            y_torso=y_torso if Pi is not None else None,
            Pi=Pi,
            L_e=getattr(self, "L_e", None),
            omega_e=getattr(self, "omega_e", None),
            omega_t=getattr(self, "omega_t", None),
        )
        loss, _ = self.combiner(terms)
        return loss

    def uncertainty_parameters(self):
        return self.combiner.parameters()

    def anneal_epoch(self, epoch: int, total_epochs: int) -> None:
        """Llamar una vez por época de ENTRENAMIENTO. No-op sin física."""
        if self.physics is None:
            return
        self.physics.coupling.anneal_tau(epoch, total_epochs)
        self.physics.anneal_warmup(epoch, total_epochs)
