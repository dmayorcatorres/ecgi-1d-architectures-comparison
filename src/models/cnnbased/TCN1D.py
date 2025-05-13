import math
import torch
import torch.nn as nn
from typing import List

__all__ = ["TCN1D", "TemporalBlock"]

# ──────────────────────────────────────────────────────────────────────────────
#  Bloque residual con convolución causal dilatada 1‑D  (similar a WaveNet/TCN)
# ──────────────────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """Bloque básico TCN: Conv1d(→BN→ReLU→Dropout) × 2 + residual"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Acolchado causal para mantener la longitud y la causalidad
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Proyección si cambian los canales
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 1 (causal: recortar cola)
        y = self.conv1(x)
        y = y[..., : x.shape[-1]]
        y = self.activation(self.bn1(y))
        y = self.dropout(y)
        # Conv 2
        y = self.conv2(y)
        y = y[..., : x.shape[-1]]
        y = self.activation(self.bn2(y))
        y = self.dropout(y)
        # Residual
        return y + self.downsample(x)


# ── TCN completo para clasificación / regresión paso-a-paso ───────────────
class TCN1D(nn.Module):
    """
    TCN1D – Temporal Convolutional Network con bloques dilatados.
      * Entrada : (B, C_in, T)
      * Salida  : (B, C_out, T)        (misma longitud temporal)
    """

    def __init__(
        self,
        time_steps: int,              # ← ¡AHORA LO ACEPTA!
        input_channels: int,
        output_channels: int,
        n_blocks: int = 6,
        n_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs                      # ← ignora cualquier otro argumento sobrante
    ) -> None:
        """
        `time_steps` se acepta solo para compatibilidad con ModelFactory.
        El TCN conserva la longitud temporal automáticamente, por lo que
        el valor no se utiliza internamente.
        """
        super().__init__()
        layers: List[nn.Module] = []
        in_c = input_channels
        for i in range(n_blocks):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_c,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_c = n_filters
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(n_filters, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        return self.head(y)
