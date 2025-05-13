import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DisjointLSTM(nn.Module):
    """
    Modelo LSTM basado en bloques secuenciales, inspirado en la arquitectura DisjointCNN.
    - Captura dependencias temporales a múltiples escalas mediante capas LSTM encadenadas.
    - Mantiene la resolución temporal y canales de entrada/salida.
    - Incluye normalización y opción bidireccional.
    
    Parámetros:
      - input_channels: Canales de entrada (ej: 163 en ECG)
      - time_steps: Longitud de la secuencia temporal
      - output_channels: Canales de salida (ej: 502 en EMG)
      - hidden_size: Tamaño oculto de las LSTM (default=32)
      - bidirectional: Usar LSTM bidireccional (default=False)
      - residual: Conexiones residuales entre etapas (default=True)
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 32,
                 bidirectional: bool = False,
                 residual: bool = True):
        super().__init__()
        
        self.residual = residual
        self.num_directions = 2 if bidirectional else 1
        
        # Capa inicial de proyección
        self.init_conv = nn.Conv1d(input_channels, hidden_size, kernel_size=1)
        
        # Bloques LSTM (4 etapas)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            ) for _ in range(4)
        ])
        
        # Capas lineales para bidireccional
        if bidirectional:
            self.adapters = nn.ModuleList([
                nn.Linear(hidden_size * 2, hidden_size) for _ in range(4)
            ])
        
        # Normalización después de cada LSTM
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(4)
        ])
        
        # Capa final de mapeo
        self.final_conv = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C_in, T)
        residual = x
        
        # Proyección inicial
        x = self.init_conv(x)  # (B, H, T)
        x = x.permute(0, 2, 1)  # (B, T, H)
        
        # Procesamiento por etapas
        for i, lstm in enumerate(self.lstm_layers):
            x_out, _ = lstm(x)  # (B, T, H * num_directions)
            
            # Manejo bidireccional
            if self.num_directions == 2:
                x_out = self.adapters[i](x_out)  # (B, T, H)
            
            # Normalización + activación
            x_out = x_out.permute(0, 2, 1)  # (B, H, T)
            x_out = self.bn_layers[i](x_out)
            x_out = F.gelu(x_out)
            
            # Conexión residual
            if self.residual and i > 0:
                x_out = x_out + x
                
            x = x_out.permute(0, 2, 1)  # Prepara para siguiente LSTM
        
        # Capa final y residual global
        x = x_out + residual if self.residual else x_out
        return self.final_conv(x)  # (B, C_out, T)