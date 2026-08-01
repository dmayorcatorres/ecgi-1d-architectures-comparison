import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .SqueezeExciteBlock import SqueezeExciteBlock

class ConvSEBlock(nn.Module):
    """
    Bloque genérico de Conv1d + BatchNorm + ReLU + (opcional) Squeeze-and-Excitation.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 se_ratio: int = 16,
                 use_se: bool = True,
                 padding: Optional[str] = 'same',
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExciteBlock(out_channels, ratio=se_ratio)
        else:
            self.se = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.use_se and self.se is not None:
            x = self.se(x)
        return x
