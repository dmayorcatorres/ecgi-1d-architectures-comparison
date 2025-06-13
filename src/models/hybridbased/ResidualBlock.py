import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class ResBlock(nn.Module):
    """
    Bloque residual simple para redes 1D.
    
    Este bloque consta de tres capas convolucionales 1D, cada una seguida de Batch Normalization y activación ReLU.
    Además, se incorpora una conexión de atajo (shortcut) para facilitar el flujo del gradiente, la cual se ajusta mediante
    una convolución 1x1 si la dimensión de entrada no coincide con la de salida o si se aplica un stride diferente de 1.
    
    Parámetros:
      - in_channels (int): Número de canales de entrada.
      - out_channels (int): Número de canales de salida.
      - stride (int, opcional): Stride a utilizar en la primera convolución (por defecto 1).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # Primera convolución: kernel de tamaño 5
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        
        # Segunda convolución: kernel de tamaño 3
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        
        # Tercera convolución: kernel de tamaño 3
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut: se ajusta la dimensión de entrada si es necesario
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)


class ConvBNReLU(nn.Module):
    """
    Sub-bloque que aplica Batch Normalization, seguida de una activación ReLU y una convolución 1D,
    utilizando padding para preservar la longitud temporal.
    
    Parámetros:
      - in_channels (int): Número de canales de entrada.
      - out_channels (int): Número de canales de salida.
      - kernel_size (int, opcional): Tamaño del kernel convolucional (por defecto 3).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.bn    = nn.BatchNorm1d(in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv  = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class ResBlock3x(nn.Module):
    """
    Bloque residual compuesto por tres sub-bloques consecutivos de la forma:
      BatchNorm → ReLU → Convolución 1D.
    
    Al final se suma la entrada (o se ajusta mediante una proyección 1x1 si es necesario) para crear la conexión residual.
    
    Parámetros:
      - in_channels (int): Número de canales de entrada.
      - out_channels (int): Número de canales de salida.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.subblock1 = ConvBNReLU(in_channels, out_channels)
        self.subblock2 = ConvBNReLU(out_channels, out_channels)
        self.subblock3 = ConvBNReLU(out_channels, out_channels)
        
        # Ajuste de la dimensión de la entrada si es necesario
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_conv(x)
        out = self.subblock1(x)
        out = self.subblock2(out)
        out = self.subblock3(out)
        out += identity
        return out

