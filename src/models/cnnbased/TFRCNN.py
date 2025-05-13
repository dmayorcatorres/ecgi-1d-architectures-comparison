import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from src.models.base.ResidualBlock import ResBlock3x

class TFRCNN(nn.Module):
    """
    TFRCNN para clasificación de series temporales, inspirado en el paper:
    'A Time-Frequency Residual Convolution Neural Network for Fault Diagnosis'
    
    Características principales:
    - Doble rama (tiempo y frecuencia)
    - Bloques residuales en ambas ramas
    - Fusión de características temporales y frecuenciales
    
    Parámetros:
      - time_steps: Pasos temporales de entrada
      - input_channels: Canales de entrada (ej. sensores)
      - output_channels: Canales de salida
      - stage1_out: Filtros iniciales (default=64)
      - stage2_out: Filtros etapa media (default=128)
      - stage3_out: Filtros etapa final (default=128)
    """
    
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 64,  # Parámetro esperado
                 stage2_out: int = 128, # Parámetro esperado
                 stage3_out: int = 128):# Parámetro esperado
        super().__init__()
        
        self.time_steps = time_steps
        self.output_channels = output_channels

        # Configuración de capas
        s1, s2, s3 = stage1_out, stage2_out, stage3_out

        # Rama Temporal (TSRCNN)
        self.time_branch = nn.Sequential(
            ResBlock3x(input_channels, s1),
            ResBlock3x(s1, s2),
            ResBlock3x(s2, s3)
        )

        # Rama Frecuencial (FSRCNN)
        self.freq_branch = nn.Sequential(
            ResBlock3x(2 * input_channels, s1),  # 2* por componentes real/imag
            ResBlock3x(s1, s2),
            ResBlock3x(s2, s3)
        )

        # Pooling y capas finales
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Como en el paper
        self.fc = nn.Linear(2 * s3, output_channels * time_steps)  # Fusión de características

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rama Temporal
        t_out = self.time_branch(x)
        
        # Rama Frecuencial
        x_freq = torch.fft.fft(x, dim=-1)  # FFT completa
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)  # Concatenar real/imag
        f_out = self.freq_branch(x_freq)

        # Pooling y fusión
        t_pool = self.global_pool(t_out).squeeze(-1)
        f_pool = self.global_pool(f_out).squeeze(-1)
        combined = torch.cat([t_pool, f_pool], dim=1)
        
        # Capa final
        out = self.fc(combined)
        return out.view(-1, self.output_channels, self.time_steps)