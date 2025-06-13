import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .ResidualBlock import ResBlock3x

class FCN(nn.Module):
    """
    FCNModel para, por ejemplo, reconstruir series temporales de EMG a ECG.
    
    Este modelo Fully Convolutional Network (FCN) utiliza bloques residuales
    y un Global Max Pooling para extraer y comprimir la representación, 
    luego expande la señal a la longitud temporal especificada en 'time_steps'
    mediante una capa fully connected.
    
    Parámetros:
      - time_steps (int): Número de pasos temporales esperados en la salida 
                          (debe coincidir con la dimensión temporal del target).
      - input_channels (int): Número de canales de la señal de entrada (ej. 221).
      - output_channels (int): Número de canales de la señal de salida (ej. 141).
      - stage1_out (int, opcional): Número de filtros en los bloques residuales (por defecto 64).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, stage1_out: int = 64):
        super().__init__()
        self.time_steps = time_steps          # Se asume que ya has reducido tu señal a 'time_steps' (p.ej. 256)
        self.output_channels = output_channels
        
        # Bloques residuales (ajusta a tu gusto la configuración)
        self.block1 = ResBlock3x(input_channels, stage1_out)
        self.block2 = ResBlock3x(stage1_out, stage1_out)
        self.block3 = ResBlock3x(stage1_out, stage1_out)
        
        # Global Max Pooling sobre la dimensión temporal (reduce T a 1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # Capa fully connected final: de (batch, stage1_out) -> (batch, output_channels * time_steps)
        self.fc = nn.Linear(stage1_out, output_channels * time_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
          x: Tensor de forma (B, input_channels, T_reducido)
             - B: batch size
             - input_channels: canales de la señal
             - T_reducido: longitud temporal tras PAA (por ejemplo, 256)
        
        Retorna:
          Tensor de forma (B, output_channels, self.time_steps)
          donde self.time_steps es el valor que se pasó al constructor.
        """
        # Pasa por los bloques residuales
        out = self.block1(x)    # (B, stage1_out, T_reducido)
        out = self.block2(out)  # (B, stage1_out, T_reducido)
        out = self.block3(out)  # (B, stage1_out, T_reducido)
        
        # Global Max Pooling => (B, stage1_out, 1)
        out = self.global_max(out)
        # Quitar la dimensión extra => (B, stage1_out)
        out = out.squeeze(-1)
        
        # Capa fully connected => (B, output_channels * time_steps)
        out = self.fc(out)
        
        # Reshape final => (B, output_channels, time_steps)
        out = out.view(-1, self.output_channels, self.time_steps)
        return out
