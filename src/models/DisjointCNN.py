import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

class DisjointBlock(nn.Module):
    def __init__(self, in_height, out_filters, temporal_kernel):
        """
        Parámetros:
          - in_height: número de variates (o “altura”) de la entrada a esta etapa.
          - out_filters: número de filtros (salida del bloque temporal).
          - temporal_kernel: tamaño del kernel que opera en la dimensión temporal (1, temporal_kernel).
        """
        super().__init__()
        self.temporal_kernel = temporal_kernel
        # Calcular padding asimétrico para preservar T
        if temporal_kernel % 2 == 0:
            self.pad_left = temporal_kernel // 2 - 1  # Ej: 8 -> 3
            self.pad_right = temporal_kernel // 2       # Ej: 8 -> 4
        else:
            self.pad_left = self.pad_right = temporal_kernel // 2

        # Convolución temporal: opera a lo largo de la dimensión temporal.
        self.temp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, 0)
        )
        self.temp_bn = nn.BatchNorm2d(out_filters)
        self.temp_act = nn.ELU(alpha=1.0)
        
        # Convolución espacial: colapsa la dimensión de variates.
        # Usamos un kernel de tamaño (in_height, 1) para cubrir toda la "altura".
        self.spatial_conv = nn.Conv2d(
            in_channels=out_filters,
            out_channels=out_filters,
            kernel_size=(in_height, 1),
            padding=0
        )
        self.spatial_bn = nn.BatchNorm2d(out_filters)
        self.spatial_act = nn.ELU(alpha=1.0)

    def forward(self, x):
        """
        Se asume que x tiene forma (B, current_height, T),
        donde current_height es el número de variates y T es la longitud temporal.
        """
        # Reestructurar para Conv2d: (B, 1, current_height, T)
        x = x.unsqueeze(1)
        # Aplicar padding asimétrico en la dimensión temporal (última dimensión)
        # El orden en F.pad es (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, (self.pad_left, self.pad_right, 0, 0))
        
        # Bloque temporal
        x = self.temp_conv(x)         # (B, out_filters, current_height, T)
        x = self.temp_bn(x)
        x = self.temp_act(x)
        
        # Bloque espacial: kernel (in_height, 1) colapsa la dimensión de variates.
        x = self.spatial_conv(x)      # (B, out_filters, 1, T)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        
        # Eliminar la dimensión "altura" (de tamaño 1) para obtener (B, out_filters, T)
        x = x.squeeze(2)
        return x

class DisjointCNN(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, num_filters: int = 32):
        """
        Parámetros:
          - input_channels: número de variates de la serie de entrada (por ejemplo, 163 para ECG).
          - time_steps: longitud temporal de la entrada (por ejemplo, 100).
          - output_channels: número de canales de salida (por ejemplo, 502 para EMG).
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        # Etapa 1: Procesa la entrada original.
        # Usa un kernel temporal mayor (8) para captar un contexto amplio.
        self.stage1 = DisjointBlock(in_height=input_channels, out_filters=num_filters, temporal_kernel=8)
        # Etapa 2:
        self.stage2 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=5)
        # Etapa 3:
        self.stage3 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=5)
        # Etapa 4:
        self.stage4 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=3)
        
        # Capa final: Convolución 1D que mapea de num_filters a output_channels, preservando T.
        self.final_conv = nn.Conv1d(in_channels=num_filters, out_channels=output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entrada:
          x: Tensor de forma (B, input_channels, time_steps)
        Salida:
          Tensor de forma (B, output_channels, time_steps)
        """
        x = self.stage1(x)  # (B, num_filters, T)
        x = self.stage2(x)  # (B, num_filters, T)
        x = self.stage3(x)  # (B, num_filters, T)
        x = self.stage4(x)  # (B, num_filters, T)
        # Capa final 1D: mapea de num_filters a output_channels sin alterar T
        fc_out = self.final_conv(x)  # (B, output_channels, T)
        return fc_out
    