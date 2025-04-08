import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from src.models.base.DisjointBase import DisjointBase


class DisjointCNN(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, num_filters: int = 32):
        """
        Modelo CNN basado en bloques disjoint utilizando DisjointBase.
        
        Parámetros:
          - input_channels: número de variates de la serie de entrada (por ejemplo, 163 para ECG).
          - time_steps: longitud temporal de la entrada (por ejemplo, 100).
          - output_channels: número de canales de salida (por ejemplo, 502 para EMG).
          - num_filters: número de filtros a utilizar en cada bloque disjoint.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        # Etapa 1: Se procesa la entrada original.
        # Se usa un kernel temporal mayor (8) para captar un contexto amplio.
        self.stage1 = DisjointBase(in_height=input_channels, 
                                   out_filters=num_filters, 
                                   temporal_kernel=8)
        # Etapa 2: La altura de entrada pasa a ser num_filters (salida del bloque previo)
        self.stage2 = DisjointBase(in_height=num_filters, 
                                   out_filters=num_filters, 
                                   temporal_kernel=5)
        # Etapa 3:
        self.stage3 = DisjointBase(in_height=num_filters, 
                                   out_filters=num_filters, 
                                   temporal_kernel=5)
        # Etapa 4:
        self.stage4 = DisjointBase(in_height=num_filters, 
                                   out_filters=num_filters, 
                                   temporal_kernel=3)
        
        # Capa final: Convolución 1D que mapea de num_filters a output_channels sin alterar la longitud temporal.
        self.final_conv = nn.Conv1d(in_channels=num_filters, out_channels=output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
          - x: Tensor de entrada con forma (B, input_channels, time_steps)
        Retorna:
          - Tensor de salida con forma (B, output_channels, time_steps)
        """
        x = self.stage1(x)  # Salida: (B, num_filters, T)
        x = self.stage2(x)  # Salida: (B, num_filters, T)
        x = self.stage3(x)  # Salida: (B, num_filters, T)
        x = self.stage4(x)  # Salida: (B, num_filters, T)
        
        # Aplicar la capa final para mapear a output_channels.
        fc_out = self.final_conv(x)  # (B, output_channels, T)
        return fc_out