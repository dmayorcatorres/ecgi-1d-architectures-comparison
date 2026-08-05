# models/atbased/PhysFormer.py
#
# Portado desde physformer1.0.ipynb (prototipo Colab) al pipeline activo.
# Se deja fuera del port lo que dependía de geometría torso-corazón
# (PhysicsMixer, pérdidas geométricas, dataset con nodes/faces): la
# ECGDataModule actual solo expone (ecg, emg, tag), no mallas — portar esa
# parte requeriría extender el datamodule y está fuera de alcance de este
# empaquetado. Tampoco se porta TransformerTimeEncoder (variante alterna no
# usada por el Transformer1D del notebook).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodingSin(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))  # (1,D,T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, :x.size(-1)]


class RotaryPE(nn.Module):
    """RoPE para head_dim par."""
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
        inv_freq = self.base_theta ** (
            -torch.arange(0, half, dtype=torch.float32, device=device) / half
        )
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,f->tf", t, inv_freq)
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


def _alibi_slopes(n_heads: int):
    def get_slopes(n):
        def power_of_two(n):
            return 2 ** math.floor(math.log2(n))
        m = power_of_two(n)
        slopes = torch.pow(2, -torch.arange(1, m + 1, dtype=torch.float32))
        if m < n:
            extra = torch.pow(2, -torch.linspace(1, m, steps=n - m, dtype=torch.float32))
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes
    return get_slopes(n_heads)


class AlibiBias(nn.Module):
    """Bias simétrico por |i-j| (no causal), por cabeza."""
    def __init__(self, n_heads: int):
        super().__init__()
        slopes = _alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))
        self.register_buffer("_cached_T", torch.tensor(0), persistent=False)
        self.register_buffer("_bias", torch.empty(0), persistent=False)

    def _build_bias(self, T: int, device, dtype):
        t = torch.arange(T, device=device)
        dist = (t[None, :] - t[:, None]).abs().to(dtype)
        bias = -dist[None, None, :, :] * self.slopes.to(device=device, dtype=dtype)
        self._cached_T = torch.tensor(T, device=device)
        self._bias = bias

    def forward(self, T: int, device, dtype):
        if (self._bias.numel() == 0) or (self._cached_T.item() < T):
            self._build_bias(T, device, dtype)
        return self._bias[:, :, :T, :T]


class MSWrapper(nn.Module):
    """Pirámide temporal multiescala para módulos (B,T,D)->(B,T,D)."""
    def __init__(self, module: nn.Module, scales=(1, 2, 4)):
        super().__init__()
        self.module = module
        self.scales = scales
        self.alpha = nn.Parameter(torch.zeros(len(scales)))

    def _down(self, x_btD: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1:
            return x_btD
        return F.avg_pool1d(x_btD.transpose(1, 2), kernel_size=s, stride=s).transpose(1, 2)

    def _up(self, x_btD: torch.Tensor, T: int, s: int) -> torch.Tensor:
        if s == 1:
            return x_btD
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


class MultiScaleTemporal(nn.Module):
    """Bloque conv1d multiescala barato (depthwise+pointwise)."""
    def __init__(self, d_model: int, kernels=(5, 9, 21), dilations=(1, 2, 4),
                 dropout: float = 0.1, depthwise: bool = True):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            if depthwise:
                branch = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad,
                              dilation=d, groups=d_model, bias=False),
                    nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                    nn.GELU(),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, dilation=d, bias=True),
                    nn.GELU(),
                )
            self.branches.append(branch)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(len(kernels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        w = self.alpha.softmax(dim=0)
        y = sum(w[i] * outs[i] for i in range(len(outs)))
        return self.dropout(y)


class SelfAttention1D(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.1,
                 pe_type: str = "sin", rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.pe_type = pe_type.lower()
        self.rope = RotaryPE(self.head_dim, base_theta=rope_theta) if self.pe_type == "rope" else None
        self.alibi = AlibiBias(num_heads) if self.pe_type == "alibi" else None

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
        if self.alibi is not None:
            bias = self.alibi(T, device=x.device, dtype=x.dtype)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)

        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(ctx)


class InceptionMultiScale1D(nn.Module):
    """
    Variante Inception 1D multiescala:
      - ramas depthwise separables con kernels/dilataciones variados
      - ramas de avg-pool y (opcional) max-pool
      - concatenación + mezcla 1x1 para volver a d_model
      - SE-gating por muestra
      - normalización por canal (BN/GN/LN)
    Entrada/salida: (B, D, T)
    """
    def __init__(self, d_model: int, kernels=(7, 15, 31, 41), dilations=(1, 2, 3, 5),
                 use_maxpool: bool = True, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = (k // 2) * d
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad,
                          dilation=d, groups=d_model, bias=False),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU(),
            ))

        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.GELU(),
        )

        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.max_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
                nn.GELU(),
            )

        n_branches = len(self.branches) + 1 + (1 if use_maxpool else 0)
        self.mix = nn.Conv1d(d_model * n_branches, d_model, kernel_size=1, bias=True)

        if norm == "bn":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm == "gn":
            self.norm = nn.GroupNorm(1, d_model)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, max(d_model // 4, 1), 1), nn.GELU(),
            nn.Conv1d(max(d_model // 4, 1), d_model, 1), nn.Sigmoid(),
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
    """Bloque conv ligero (depthwise + pointwise) para sesgo inductivo local. Trabaja en (B, D, T)."""
    def __init__(self, d_model: int, kernel: int = 7, norm: str = "gn", dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad,
                            groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act = nn.GELU()
        if norm == "bn":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm == "gn":
            self.norm = nn.GroupNorm(1, d_model)
        else:
            self.norm = nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bdT: torch.Tensor) -> torch.Tensor:
        y = self.dw(x_bdT)
        y = self.pw(y)
        y = self.act(y)
        y = self.norm(y)
        return self.drop(y)


class TransformerBlock1D(nn.Module):
    """
    Bloque Transformer 1D:
      - Atención (SelfAttention1D) envuelta en pirámide multiescala (MSWrapper)
      - Convolución local (ConvSandwich1D)
      - MLP feed-forward
    I/O: (B, T, D)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 pe_type: str = "sin", rope_theta: float = 10000.0,
                 attn_ms_scales=(1, 2, 4), conv_kernel: int = 7, conv_norm: str = "gn"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        base_attn = SelfAttention1D(
            d_model=d_model, num_heads=num_heads, attn_drop=dropout,
            pe_type=pe_type, rope_theta=rope_theta,
        )
        self.attn = MSWrapper(base_attn, scales=attn_ms_scales)

        self.conv_sandwich = ConvSandwich1D(
            d_model=d_model, kernel=conv_kernel, norm=conv_norm, dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))

        xc = self.conv_sandwich(x.permute(0, 2, 1))
        x = x + xc.permute(0, 2, 1)

        x = x + self.mlp(self.ln2(x))
        return x


class PhysFormer(nn.Module):
    """
    PhysFormer (portado de physformer1.0.ipynb, sin el término de pérdida
    físico basado en geometría torso-corazón — ver nota al inicio del
    archivo). Transformer 1D con atención multiescala (pirámide de
    downsample/upsample sobre 3 escalas), frente multiescala temporal
    (conv depthwise "vanilla" o Inception-1D), y positional encoding
    intercambiable (seno fijo / RoPE / ALiBi).

    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        pe_type: str = "sin",
        rope_theta: float = 10000.0,
        use_ms_temporal: bool = True,
        ms_type: str = "vanilla",           # "vanilla" | "inception"
        ms_kernels=(3, 7, 15),
        ms_dilations=(1, 2, 4),
        attn_ms_scales=(1, 2, 4),
        use_label_concat: bool = False,
        num_classes: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.pe_type = pe_type.lower()
        self.use_ms_temporal = use_ms_temporal
        self.ms_type = ms_type.lower()

        self.use_label_concat = bool(use_label_concat)
        self.num_classes = int(num_classes or 0) if self.use_label_concat else 0
        if self.use_label_concat and self.num_classes <= 0:
            raise ValueError("num_classes debe ser > 0 cuando use_label_concat=True")

        in_ch = input_channels + (self.num_classes if self.use_label_concat else 0)

        self.input_proj = nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.pos_enc = PositionalEncodingSin(d_model, max_len=time_steps) if self.pe_type == "sin" else None

        if use_ms_temporal:
            if self.ms_type == "inception":
                self.ms_temporal = InceptionMultiScale1D(
                    d_model, kernels=(7, 15, 31, 41), dilations=(1, 2, 3, 5),
                    use_maxpool=True, norm="gn", dropout=dropout,
                )
            elif self.ms_type == "vanilla":
                self.ms_temporal = MultiScaleTemporal(
                    d_model, kernels=ms_kernels, dilations=ms_dilations,
                    dropout=dropout, depthwise=True,
                )
            else:
                raise ValueError(f"ms_type desconocido: {ms_type}")
        else:
            self.ms_temporal = None

        self.layers = nn.ModuleList([
            TransformerBlock1D(
                d_model, num_heads, dropout=dropout,
                pe_type=self.pe_type, rope_theta=rope_theta,
                attn_ms_scales=attn_ms_scales,
                conv_kernel=7, conv_norm="gn",
            )
            for _ in range(num_layers)
        ])

        self.out_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, y_cls: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_label_concat:
            if y_cls is None:
                raise ValueError("Se requiere y_cls con use_label_concat=True")
            y_oh = F.one_hot(y_cls, num_classes=self.num_classes).to(dtype=x.dtype, device=x.device)
            y_oh = y_oh.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.cat([x, y_oh], dim=1)

        x = self.input_proj(x)
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
