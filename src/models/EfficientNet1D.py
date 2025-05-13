import math
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List, Optional


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
def _round_filters(filters: int, width_mult: float, divisor: int = 8) -> int:
    """Redondea los filtros al múltiplo de `divisor` más cercano (EfficientNet rule)."""
    if not width_mult:
        return filters
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:      # evita redondeo excesivo a la baja
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_mult: float) -> int:
    """Escala la profundidad (nº de bloques) y redondea al entero más cercano."""
    return int(math.ceil(repeats * depth_mult))


# ───────────────────────────────────────────────────────────────────────────
# Bloques básicos
# ───────────────────────────────────────────────────────────────────────────
class ConvBNAct(nn.Sequential):
    """Conv1d → BatchNorm → SiLU  (activación Swish)"""
    def __init__(self, in_c, out_c, k, stride=1, groups=1):
        padding = k // 2
        super().__init__(
            nn.Conv1d(in_c, out_c, k, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_c),
            nn.SiLU(inplace=True)
        )


class SqueezeExcite1D(nn.Module):
    def __init__(self, in_c: int, se_ratio: float = 0.25):
        super().__init__()
        reduc = max(1, int(in_c * se_ratio))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_c, reduc, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(reduc, in_c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class MBConv1D(nn.Module):
    """
    Bloque MBConv (Mobile-Inverted Bottleneck) 1-D:
    expand → depthwise → SE → project (+ residual si posible)
    """
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 expansion: int,
                 k: int,
                 stride: int,
                 se_ratio: float = 0.25):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        mid_c = in_c * expansion

        layers = []
        # Expand
        if expansion != 1:
            layers.append(ConvBNAct(in_c, mid_c, 1))
        # Depthwise
        layers.append(
            ConvBNAct(mid_c, mid_c, k, stride=stride, groups=mid_c)
        )
        # SE
        layers.append(SqueezeExcite1D(mid_c, se_ratio))
        # Project
        layers.append(
            nn.Conv1d(mid_c, out_c, 1, bias=False)
        )
        layers.append(nn.BatchNorm1d(out_c))
        self.block = nn.Sequential(*layers)
        self.act = nn.Identity()   # resid swish ya aplicado

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


# ───────────────────────────────────────────────────────────────────────────
# EfficientNet-1D
# ───────────────────────────────────────────────────────────────────────────
_DEFAULT_CFG: List[Tuple[int, int, int, int, int]] = [
    # (expansion, out_c, repeats, kernel, stride)
    (1,   16, 1, 3, 1),
    (6,   24, 2, 3, 2),
    (6,   40, 2, 5, 2),
    (6,   80, 3, 3, 2),
    (6,  112, 3, 5, 1),
    (6,  192, 4, 5, 2),
    (6,  320, 1, 3, 1),
]


class EfficientNet1D(nn.Module):
    """
    *input*:  (B, C_in, T)      – C_in = canales de señal (p.ej. 163)
    *output*: (B, C_out, T_out) – clasificación/ logits paso-a-paso.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.out_ch = _round_filters(1280, width_mult)  # canal de la capa final

        layers = []
        # stem
        layers.append(
            ConvBNAct(input_channels,
                      _round_filters(32, width_mult),
                      k=3, stride=2)
        )
        in_c = _round_filters(32, width_mult)

        # blocks
        for expansion, c, r, k, s in _DEFAULT_CFG:
            out_c = _round_filters(c, width_mult)
            repeats = _round_repeats(r, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers.append(
                    MBConv1D(in_c, out_c, expansion, k, stride)
                )
                in_c = out_c

        # head conv 1×1
        layers.append(ConvBNAct(in_c, self.out_ch, k=1))

        self.features = nn.Sequential(*layers)

        # proyector final 1×1 → logits paso-a-paso
        self.classifier = nn.Conv1d(self.out_ch, output_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Devuelve logits por paso temporal.
        T_out = ceil(T_in / 2^n_strides)
        """
        x = self.features(x)
        x = self.dropout(x)
        logits = self.classifier(x)       # (B, output_channels, T_out)
        return logits


# ───────────────────────────────────────────────────────────────────────────
# Prueba rápida
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, C_in, T = 4, 1, 1024
    model = EfficientNet1D(
        input_channels=C_in,
        output_channels=5,     # p.ej. 5 clases
        width_mult=0.75,       # escala “b0 ≈ 1.0”
        depth_mult=0.75,
    )
    x = torch.randn(B, C_in, T)
    y = model(x)
    print("Output shape:", y.shape)   # -> (B, 5, T_out)
