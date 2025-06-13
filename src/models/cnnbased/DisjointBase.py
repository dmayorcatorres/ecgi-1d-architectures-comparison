import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

########################################
# Clase Base: DisjointBase
########################################
class DisjointBase(nn.Module):
    def __init__(self, in_height: int, out_filters: int, temporal_kernel: int, activation: nn.Module = nn.ELU):
        """
        Bloque básico que opera en dos etapas:
          1. Convolución temporal (con padding asimétrico) para extraer características locales en el tiempo.
          2. Convolución espacial para colapsar la dimensión de variates.
        
        Parámetros:
          - in_height: número de variates (o “altura”) de la entrada.
          - out_filters: número de filtros de salida del bloque temporal.
          - temporal_kernel: tamaño del kernel que opera en la dimensión temporal.
          - activation: función de activación (por defecto nn.ELU).
        """
        super().__init__()
        self.temporal_kernel = temporal_kernel
        
        # Calcular padding asimétrico para preservar la dimensión temporal T.
        if temporal_kernel % 2 == 0:
            self.pad_left = temporal_kernel // 2 - 1
            self.pad_right = temporal_kernel // 2
        else:
            self.pad_left = self.pad_right = temporal_kernel // 2

        # Bloque temporal: la convolución opera sobre la dimensión temporal.
        self.temp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, 0)
        )
        self.temp_bn = nn.BatchNorm2d(out_filters)
        self.temp_act = activation(alpha=1.0) if hasattr(activation, '__call__') else activation()

        # Bloque espacial: se colapsa la dimensión de variates utilizando un kernel que cubre toda la altura.
        self.spatial_conv = nn.Conv2d(
            in_channels=out_filters,
            out_channels=out_filters,
            kernel_size=(in_height, 1),
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(out_filters)
        self.spatial_act = activation(alpha=1.0) if hasattr(activation, '__call__') else activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
          - x: tensor de forma (B, in_height, T)
        Retorna:
          - Tensor de forma (B, out_filters, T)
        """
        # Expandir dimensión para poder aplicar Conv2d: (B, 1, in_height, T)
        x = x.unsqueeze(1)
        # Aplicar padding asimétrico en la dimensión temporal
        x = F.pad(x, (self.pad_left, self.pad_right, 0, 0))
        
        # Bloque temporal
        x = self.temp_conv(x)         # (B, out_filters, in_height, T)
        x = self.temp_bn(x)
        x = self.temp_act(x)
        
        # Bloque espacial (colapsa la dimensión "in_height")
        x = self.spatial_conv(x)      # (B, out_filters, 1, T)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        
        # Eliminar la dimensión de tamaño 1 (la "altura")
        x = x.squeeze(2)              # (B, out_filters, T)
        return x