import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.ResidualBlock import ResBlock3x  # Se asume que este módulo define el bloque residual de 3 subcapas

class ResNetCNN(nn.Module):
    """
    ResNetCNN – Modelo de red residual para clasificación de series temporales.

    Esta arquitectura procesa la entrada en forma (B, T, C) mediante:
      1. Una capa inicial de convolución 1D que transforma la entrada a conv_in_out canales.
      2. Una secuencia de bloques residuales (ResBlock3x) que extraen características profundas.
      3. Un Global Average Pooling que reduce la dimensión temporal a 1 para obtener una representación global.
      4. Una capa fully connected que mapea esta representación a la dimensión de salida.
      5. Finalmente, la predicción (de forma (B, output_channels)) se replica a lo largo de la dimensión temporal,
         produciendo una salida de forma (B, T, output_channels).

    Parámetros:
      - time_steps (int): Número de pasos temporales (T) de la entrada.
      - input_channels (int): Número de canales de la entrada (C) (ej. 163 para ECG).
      - output_channels (int): Número de clases o canales de salida (ej. 502 para EMG).
      - num_blocks (int, opcional): Número de bloques residuales a utilizar (por defecto 1).
      - conv_in_out (int, opcional): Número de filtros en la capa inicial y en los bloques residuales (por defecto 64).
      - kernel_size_in (int, opcional): Tamaño del kernel en la capa inicial (por defecto 7).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 num_blocks: int = 1, conv_in_out: int = 64, kernel_size_in: int = 7) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels

        # Capa inicial: transforma la entrada a conv_in_out canales.
        # La entrada se espera en forma (B, C, T).
        self.conv_in = nn.Conv1d(in_channels=input_channels,
                                 out_channels=conv_in_out,
                                 kernel_size=kernel_size_in,
                                 padding=kernel_size_in // 2)
        self.bn_in = nn.BatchNorm1d(conv_in_out)
        self.relu = nn.ReLU(inplace=True)
        
        # Bloques residuales: se emplean num_blocks bloques residuales.
        # Cada bloque preserva la dimensión de salida (conv_in_out).
        self.res_blocks = nn.Sequential(*(ResBlock3x(conv_in_out, conv_in_out) for _ in range(num_blocks)))
        
        # Global Average Pooling: reduce la dimensión temporal a 1.
        # Salida: (B, conv_in_out, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Capa fully connected: mapea la representación global a (B, output_channels).
        self.fc = nn.Linear(conv_in_out, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.

        Parámetros:
          - x: Tensor de entrada de forma (B, T, C), donde:
              B = tamaño del batch,
              T = número de pasos temporales,
              C = número de canales (input_channels).
        
        Retorna:
          - Tensor de salida de forma (B, T, output_channels)
        """
        # Reorganizar la entrada a (B, C, T) para las operaciones Conv1d.
        x = x.permute(0, 2, 1)
        # Capa inicial: conv → BN → ReLU
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        # Aplicar los bloques residuales
        x = self.res_blocks(x)
        
        # Global Average Pooling en la dimensión temporal: (B, conv_in_out, 1)
        x = self.gap(x).squeeze(-1)  # (B, conv_in_out)
        
        # Capa fully connected: (B, conv_in_out) → (B, output_channels)
        x = self.fc(x)
        
        # Replicar la predicción a lo largo de la dimensión temporal para obtener (B, T, output_channels)
        x = x.unsqueeze(1).repeat(1, self.time_steps, 1)
        return x