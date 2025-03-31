import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ZhaoCNN(nn.Module):
    """
    ZhaoCNN – Modelo de CNN multicanal para clasificación de series temporales.

    Esta arquitectura procesa una entrada de forma (B, C, T) mediante:
      1. Una primera etapa de convolución 1D, seguida de Batch Normalization, activación ReLU y max pooling.
      2. Una segunda etapa de convolución 1D, seguida de Batch Normalization, activación ReLU y max pooling.
      3. Opcionalmente, una tercera etapa de convolución 1D, seguida de Batch Normalization, activación ReLU y max pooling.
      4. Se aplana la representación y se utiliza una capa fully connected para generar una predicción única por muestra.
      5. Finalmente, la predicción (de forma (B, output_channels)) se replica a lo largo de la dimensión temporal,
         produciendo una salida final de forma (B, output_channels, T).

    Parámetros:
      - time_steps (int): Número de pasos temporales de la entrada (T).
      - input_channels (int): Número de canales de la entrada (C) (e.g., 163 para ECG).
      - output_channels (int): Número de clases o canales de salida (e.g., 502 para EMG).
      - stage1_out (int, opcional): Número de filtros de la primera etapa (por defecto 128).
      - kernel_size_stage1 (int, opcional): Tamaño del kernel en la primera etapa (por defecto 5).
      - stage2_out (int, opcional): Número de filtros de la segunda etapa (por defecto 64).
      - kernel_size_stage2 (int, opcional): Tamaño del kernel en la segunda etapa (por defecto 3).
      - stage3_out (int, opcional): Número de filtros de la tercera etapa (por defecto None, es decir, se usan solo dos etapas).
      - kernel_size_stage3 (int, opcional): Tamaño del kernel en la tercera etapa (por defecto 3).
      - pool_kernel (int, opcional): Tamaño del kernel de max pooling (por defecto 2).

    Se espera que la entrada tenga forma (B, input_channels, T) y la salida generada tenga forma (B, output_channels, T).
    """
    def __init__(self, 
                 time_steps: int, 
                 input_channels: int, 
                 output_channels: int,
                 stage1_out: int = 128, 
                 kernel_size_stage1: int = 5,
                 stage2_out: int = 64, 
                 kernel_size_stage2: int = 3,
                 stage3_out: int = None, 
                 kernel_size_stage3: int = 3,
                 pool_kernel: int = 2) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels
        
        # Etapa 1
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=stage1_out,
                               kernel_size=kernel_size_stage1,
                               padding=kernel_size_stage1 // 2)
        self.bn1 = nn.BatchNorm1d(stage1_out)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_kernel)
        
        # Etapa 2
        self.conv2 = nn.Conv1d(in_channels=stage1_out,
                               out_channels=stage2_out,
                               kernel_size=kernel_size_stage2,
                               padding=kernel_size_stage2 // 2)
        self.bn2 = nn.BatchNorm1d(stage2_out)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_kernel)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Etapa 3 (opcional)
        if stage3_out is not None:
            self.conv3 = nn.Conv1d(in_channels=stage2_out,
                                   out_channels=stage3_out,
                                   kernel_size=kernel_size_stage3,
                                   padding=kernel_size_stage3 // 2)
            self.bn3 = nn.BatchNorm1d(stage3_out)
            self.pool3 = nn.MaxPool1d(kernel_size=pool_kernel)
            self.final_channels = stage3_out
            self.num_stages = 3
        else:
            self.final_channels = stage2_out
            self.num_stages = 2
        
        # Calcular la longitud temporal final tras las operaciones de pooling.
        self.final_length = time_steps // (pool_kernel ** self.num_stages)
        
        # Capa fully connected: mapea la representación a (output_channels).
        # La entrada es un vector de tamaño (final_channels * final_length).
        self.fc = nn.Linear(self.final_channels * self.final_length, output_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Entrada:
          - x: Tensor de forma (B, C, T), donde:
              B = tamaño del batch,
              C = número de canales de entrada,
              T = longitud temporal.
        
        Retorna:
          - Tensor de forma (B, output_channels, T)
        """
        # Etapa 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Etapa 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Etapa 3, si está definida
        if self.num_stages == 3:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.pool3(x)
        
        # Aplanar la salida: (B, final_channels * final_length)
        x = x.view(x.size(0), -1)
        
        # Capa fully connected: (B, final_channels * final_length) → (B, output_channels)
        x = self.fc(x)
        
        # Replicar la predicción a lo largo de la dimensión temporal para obtener (B, output_channels, T)
        x = x.unsqueeze(2).repeat(1, 1, self.time_steps)
        return x
