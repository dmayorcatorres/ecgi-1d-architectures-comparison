import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List


class MultiScaleCNN(nn.Module):
    """Modelo MultiScale-CNN con ramas paralelas.
    
    *MODIFICADO* para incluir GAP + FC final.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 kernel_sizes: List[int] = [3, 5, 7], num_filters: int = 64) -> None:
        super().__init__()
        self.time_steps = time_steps

        # Ramas paralelas con distintos kernel_sizes
        self.branches = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(num_filters * len(kernel_sizes))

        # En lugar de conv_proj final, usaremos GAP + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, C, T)

        branch_outputs = [F.relu(conv(x)) for conv in self.branches]
        x_cat = torch.cat(branch_outputs, dim=1)   # (B, num_branches*num_filters, T)
        x_cat = self.bn(x_cat)

        # GAP + FC
        x_cat = self.gap(x_cat).squeeze(-1)  # (B, num_branches*num_filters)
        x_cat = self.fc(x_cat)               # (B, output_channels)

        # Repetir en el tiempo
        x_cat = x_cat.unsqueeze(1).repeat(1, self.time_steps, 1)
        return x_cat
