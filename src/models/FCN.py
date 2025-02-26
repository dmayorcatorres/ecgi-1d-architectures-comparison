import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from src.models.ResidualBlock import ResBlock3x

class FCN(nn.Module):
    """
    FCNModel para clasificación de series temporales.
    
    Este modelo Fully Convolutional Network (FCN) utiliza tres bloques residuales
    (cada uno implementado con ResBlock3x) para extraer características de la señal de entrada.
    Posteriormente, se aplica un Global Max Pooling sobre la dimensión temporal para obtener
    una representación global, que es luego expandida a través de una capa fully connected para
    generar la salida final con la misma longitud temporal que la entrada.
    
    Parámetros:
      - time_steps (int): Número de pasos temporales en la señal de salida.
      - input_channels (int): Número de canales de la señal de entrada (por ejemplo, 163 para ECG).
      - output_channels (int): Número de clases o canales de salida (por ejemplo, 502 para EMG).
      - stage1_out (int, opcional): Número de filtros de salida del primer bloque residual (por defecto 64).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, stage1_out: int = 64):
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels
        
        # Definición de tres bloques residuales
        self.block1 = ResBlock3x(input_channels, stage1_out)
        self.block2 = ResBlock3x(stage1_out, stage1_out)
        self.block3 = ResBlock3x(stage1_out, stage1_out)
        
        # Global Max Pooling sobre la dimensión temporal (reduce T a 1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # Capa fully connected: expande la representación (de tamaño stage1_out)
        # a un vector de tamaño (output_channels * time_steps)
        self.fc = nn.Linear(stage1_out, output_channels * time_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Parámetros:
          x: Tensor de forma (B, input_channels, time_steps)
             - B: batch size.
             - input_channels: Número de canales de entrada.
             - time_steps: Longitud temporal de la señal.
        
        Retorna:
          Tensor de forma (B, output_channels, time_steps)
        """
        # Aplicar bloques residuales
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        # Global Max Pooling en la dimensión temporal: (B, stage1_out, 1)
        out = self.global_max(out)
        out = out.squeeze(-1)  # (B, stage1_out)
        
        # Capa fully connected: mapea a (B, output_channels * time_steps)
        out = self.fc(out)
        
        # Reestructurar la salida a (B, output_channels, time_steps)
        out = out.view(-1, self.output_channels, self.time_steps)
        return out
