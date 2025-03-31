import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MultiScaleCNN(nn.Module):
    """Modelo MultiScale-CNN con ramas paralelas, modificado para aceptar parámetros específicos para cada rama.
    
    Se espera que se reciban tres valores:
      - stage1_out: número de filtros para la primera rama (kernel_size 3)
      - stage2_out: número de filtros para la segunda rama (kernel_size 5)
      - stage3_out: número de filtros para la tercera rama (kernel_size 7)
    
    Además, se incluye un bloque GAP (Global Average Pooling) seguido de una capa Fully Connected.
    
    La red espera una entrada de forma (B, input_channels, time_steps)
    y produce una salida de forma (B, output_channels, time_steps).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int, stage2_out: int, stage3_out: int,
                 kernel_sizes: List[int] = [3, 5, 7]) -> None:
        super().__init__()
        self.time_steps = time_steps

        # Se crean ramas paralelas, asignando a cada rama el número de filtros correspondiente.
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels=input_channels, out_channels=out_channels,
                      kernel_size=k, padding=k // 2)
            for k, out_channels in zip(kernel_sizes, [stage1_out, stage2_out, stage3_out])
        ])
        # El total de filtros es la suma de los de cada rama.
        total_filters = stage1_out + stage2_out + stage3_out
        self.bn = nn.BatchNorm1d(total_filters)

        # GAP seguido de una capa Fully Connected para mapear a output_channels.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(total_filters, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Se asume que la entrada ya viene en forma (B, input_channels, time_steps)
        branch_outputs = [F.relu(conv(x)) for conv in self.branches]
        # Se concatenan las salidas a lo largo del canal.
        x_cat = torch.cat(branch_outputs, dim=1)  # (B, total_filters, time_steps)
        x_cat = self.bn(x_cat)

        # Se aplica GAP para reducir la dimensión temporal a 1 y luego se aplana.
        x_gap = self.gap(x_cat).squeeze(-1)  # (B, total_filters)
        x_fc = self.fc(x_gap)               # (B, output_channels)

        # Se repite el resultado a lo largo de la dimensión temporal para obtener (B, output_channels, time_steps)
        x_out = x_fc.unsqueeze(2).repeat(1, 1, self.time_steps)
        return x_out
