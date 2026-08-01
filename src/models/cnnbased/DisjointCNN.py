import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from models.base.DisjointBase import DisjointBase


class DisjointCNN(nn.Module):
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        stage1_out: int = 32,
        ch_decay: float = 0.75,     # ← nuevo: factor de decaimiento por bloque (0.5–0.8 típico)
        min_channels: int = 32      # ← umbral mínimo para no colapsar capacidad
    ):
        """
        CNN con bloques disjoint. Como no hay downsampling temporal, usamos
        una pirámide de canales decreciente para reducir memoria.

        Parámetros:
          - input_channels: nº de variables de entrada (e.g., 192).
          - time_steps: longitud temporal de la entrada (no se altera).
          - output_channels: nº de canales de salida (e.g., 599).
          - stage1_out: canales del primer bloque.
          - ch_decay: factor multiplicativo por bloque (p.ej., 0.75).
          - min_channels: mínimo de canales por bloque.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Secuencia de canales decreciente
        c1 = int(stage1_out)
        c2 = max(min_channels, int(round(c1 * ch_decay)))
        c3 = max(min_channels, int(round(c2 * ch_decay)))
        c4 = max(min_channels, int(round(c3 * ch_decay)))
        # Garantía de no-creciente por si redondeos:
        c2 = min(c2, c1); c3 = min(c3, c2); c4 = min(c4, c3)

        # Bloques
        self.stage1 = DisjointBase(in_height=input_channels, out_filters=c1, temporal_kernel=8)
        self.stage2 = DisjointBase(in_height=c1,             out_filters=c2, temporal_kernel=5)
        self.stage3 = DisjointBase(in_height=c2,             out_filters=c3, temporal_kernel=5)
        self.stage4 = DisjointBase(in_height=c3,             out_filters=c4, temporal_kernel=3)

        # Capa final: del último nº de canales (c4) a output_channels
        self.final_conv = nn.Conv1d(in_channels=c4, out_channels=output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_channels, T) → salida: (B, output_channels, T)
        """
        x = self.stage1(x)  # (B, c1, T)
        x = self.stage2(x)  # (B, c2, T)
        x = self.stage3(x)  # (B, c3, T)
        x = self.stage4(x)  # (B, c4, T)
        return self.final_conv(x)
