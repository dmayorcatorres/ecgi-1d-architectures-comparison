import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from models.base.ResidualBlock import ResBlock3x

class ResNet(nn.Module):
    """
    Modelo ResNetCNN para clasificación de series temporales.

    Parámetros:
      - time_steps (int): Número de pasos temporales de la salida.
      - input_channels (int): Número de canales de la señal de entrada.
      - output_channels (int): Número de canales de la salida.
      - stage1_out_resnet (int): Filtros del primer bloque (default=64).
      - stage2_out_resnet (int): Filtros del segundo bloque (default=128).
      - stage3_out_resnet (int): Filtros del tercer bloque (default=128).
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 64,
                 stage2_out: int = 128,
                 stage3_out: int = 128):
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels

        # Usamos directamente los nombres que viene de Optuna
        s1 = stage1_out
        s2 = stage2_out
        s3 = stage3_out

        # Tres bloques residuales
        self.block1 = ResBlock3x(input_channels, s1)
        self.block2 = ResBlock3x(s1, s2)
        self.block3 = ResBlock3x(s2, s3)

        # Pooling global y FC para regenerar toda la secuencia
        self.global_max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(s3, output_channels * time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.global_max(out)    # (B, s3, 1)
        out = out.squeeze(-1)         # (B, s3)
        out = self.fc(out)            # (B, output_channels * time_steps)
        return out.view(-1, self.output_channels, self.time_steps)
