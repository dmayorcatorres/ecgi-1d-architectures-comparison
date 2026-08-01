import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.DisjointBase import DisjointBase

class ResidualDisjointBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int):
        """
        Bloque residual sencillo que utiliza la base DisjointBase.
        
        Parámetros:
          - in_channels: número de canales de entrada (se utiliza como 'in_height' en DisjointBase).
          - out_channels: número de filtros de salida que se desea.
          - temporal_kernel: tamaño del kernel temporal para el bloque disjoint.
        """
        super().__init__()
        self.block = DisjointBase(in_height=in_channels, out_filters=out_channels, temporal_kernel=temporal_kernel)
        # Proyección en la conexión residual si las dimensiones no coinciden.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)  # Procesa la entrada usando la base disjoint.
        shortcut = self.shortcut(x)  # Proyecta la vía residual si es necesario.
        return F.relu(out + shortcut)


class DisjointResNet(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, stage1_out: int = 32):
        """
        DisjointResNet para reconstrucción (o clasificación) que utiliza la base DisjointBase
        para construir bloques residuales de forma modular.
        
        Parámetros:
          - time_steps: longitud temporal de la serie.
          - input_channels: número de canales de entrada (p.ej., sensores ECG).
          - output_channels: número de canales de salida.
          - stage1_out: número de filtros base para los bloques residuales.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Bloque inicial: extrae características básicas usando DisjointBase.
        self.initial_block = DisjointBase(in_height=input_channels, 
                                          out_filters=stage1_out, 
                                          temporal_kernel=8)
        
        # Secuencia de bloques residuales sencillos.
        # Aquí se utilizan, por ejemplo, dos bloques residuales con kernel temporal 5.
        self.res_blocks = nn.Sequential(
            ResidualDisjointBlock(in_channels=stage1_out, out_channels=stage1_out, temporal_kernel=5),
            ResidualDisjointBlock(in_channels=stage1_out, out_channels=stage1_out, temporal_kernel=5)
        )
        
        # Capa final: se proyecta la salida al número de canales deseado sin alterar la longitud temporal.
        self.final_conv = nn.Conv1d(stage1_out, output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entrada:
          - x: Tensor de forma (B, input_channels, time_steps)
        Salida:
          - Tensor de forma (B, output_channels, time_steps)
        """
        out = self.initial_block(x)    # (B, stage1_out, T)
        out = self.res_blocks(out)     # (B, stage1_out, T)
        out = self.final_conv(out)     # (B, output_channels, T)
        return out