import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from src.models.ResidualBlock import ResBlock3x

class ResNet(nn.Module):
    """
    Modelo ResNetCNN para clasificación de series temporales.
    
    Este modelo utiliza tres bloques residuales (ResBlock3x) para extraer características locales
    de la señal de entrada. Posteriormente, se aplica un Global Max Pooling sobre la dimensión temporal
    para obtener una representación compacta, la cual se expande mediante una capa fully connected para
    generar la salida final con la misma longitud temporal que la entrada.
    
    Parámetros:
      - time_steps (int): Número de pasos temporales de la salida.
      - input_channels (int): Número de canales de la señal de entrada (ej. 163 para ECG).
      - output_channels (int): Número de canales de la salida (ej. 502 para EMG).
      - stage1_out (int, opcional): Número de filtros de salida del primer bloque residual (por defecto 64).
      - stage2_out (int, opcional): Número de filtros del segundo bloque residual (por defecto 128).
      - stage3_out (int, opcional): Número de filtros del tercer bloque residual (por defecto 128).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 64, stage2_out: int = 128, stage3_out: int = 128):
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels
        
        # Bloque 1: de input_channels a stage1_out
        self.block1 = ResBlock3x(input_channels, stage1_out)
        # Bloque 2: de stage1_out a stage2_out
        self.block2 = ResBlock3x(stage1_out, stage2_out)
        # Bloque 3: de stage2_out a stage3_out
        self.block3 = ResBlock3x(stage2_out, stage3_out)
        
        # Global Max Pooling en la dimensión temporal: (B, stage3_out, 1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # Capa fully connected para mapear la representación compacta a
        # (output_channels * time_steps)
        self.fc = nn.Linear(stage3_out, output_channels * time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Entrada:
          x: Tensor de forma (B, input_channels, time_steps)
              B: tamaño del batch.
              input_channels: número de canales de entrada.
              time_steps: longitud temporal.
        
        Salida:
          Tensor de forma (B, output_channels, time_steps)
        """
        # Aplicar los bloques residuales
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        # Global Max Pooling sobre la dimensión temporal (reduce T a 1)
        out = self.global_max(out)    # (B, stage3_out, 1)
        out = out.squeeze(-1)         # (B, stage3_out)
        
        # Capa fully connected: mapea a (B, output_channels * time_steps)
        out = self.fc(out)
        # Reestructurar la salida a (B, output_channels, time_steps)
        out = out.view(-1, self.output_channels, self.time_steps)
        return out