import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
      - time_steps: longitud de la secuencia temporal
      - input_channels: canales de entrada (por ej. 163 en ECG)
      - output_channels: canales de salida (por ej. 502 en EMG)
      - hidden_size: tamaño oculto de las LSTM (default=32)
      - bidirectional: usar LSTM bidireccional (default=False)
      - residual: incluir conexiones residuales (default=True)
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
        self.hidden_size = hidden_size
        
        # 1) Proyección inicial de los canales de entrada a 'hidden_size'
        self.init_conv = nn.Conv1d(input_channels, hidden_size, kernel_size=1)
        
        # 2) Cuatro capas LSTM encadenadas
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            ) for _ in range(4)
        ])
        
        # 3) Si es bidireccional, agregar capa lineal que reduzca (H*2) → H
        if bidirectional:
            self.adapters = nn.ModuleList([
                nn.Linear(hidden_size * 2, hidden_size) for _ in range(4)
            ])
        
        # 4) Normalización tras cada bloque LSTM (en espacio [B, H, T])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(4)
        ])
        
        # 5) Capa final que mapea de 'hidden_size' → output_channels
        self.final_conv = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
          - x: tensor de entrada con forma (B, input_channels, time_steps)
        Retorna:
          - tensor de salida con forma (B, output_channels, time_steps)
        """
        # Guardamos la proyección inicial (B, H, T) para la residual “global”
        proj = self.init_conv(x)  # (B, hidden_size, T)
        
        # Pasamos esa proyección al formato que acepta la LSTM: (B, T, H)
        x = proj.permute(0, 2, 1)  # (B, T, hidden_size)

        # Recorremos cada una de las 4 etapas
        for i, lstm in enumerate(self.lstm_layers):
            # 1) Entrada a la LSTM: x tiene shape (B, T, hidden_size)
            x_pre = x  # (B, T, hidden_size), para posible residual interno
            
            # 2) Salida de LSTM: (B, T, hidden_size * num_directions)
            x_out, _ = lstm(x_pre)
            
            # 3) Si es bidireccional, reducimos con 'adapter'
            if self.num_directions == 2:
                # x_out reshape: (B, T, hidden_size*2) → (B, T, hidden_size)
                x_out = self.adapters[i](x_out)
            
            # 4) Normalización + activación:
            #    Primero permutamos a (B, H, T) para BatchNorm1d
            x_out = x_out.permute(0, 2, 1)        # (B, hidden_size, T)
            x_out = self.bn_layers[i](x_out)     # BatchNorm sobre dimensión H
            x_out = F.gelu(x_out)                # (B, hidden_size, T)
            
            # 5) Residual interno: sumar con la entrada de esta etapa
            #    'x_pre' estaba en (B, T, H) → permutamos a (B, H, T)
            if self.residual and i > 0:
                x_out = x_out + x_pre.permute(0, 2, 1)  # Ambas formas (B, H, T)
            
            # 6) Preparamos la salida para la siguiente LSTM
            x = x_out.permute(0, 2, 1)  # (B, T, hidden_size)
        
        # Al salir del bucle, x_out tiene forma (B, H, T)
        # Residual “global”: sumar con la proyección inicial (proj)
        if self.residual:
            x_out = x_out + proj  # Ambas formas (B, hidden_size, T)
        
        # Capa final que mapea a los canales de salida
        return self.final_conv(x_out)  # (B, output_channels, T)
