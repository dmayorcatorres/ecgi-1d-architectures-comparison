# src/models/mlp_basic.py

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP para reconstrucción de series temporales con múltiples etapas configurables
    
    Parámetros:
      - time_steps (int): Pasos temporales de salida
      - input_channels (int): Canales de entrada
      - output_channels (int): Canales de salida
      - stage1_out (int): Neuronas primera capa oculta (default=500)
      - stage2_out (int): Neuronas segunda capa oculta (default=500)
      - stage3_out (int): Neuronas tercera capa oculta (default=500)
      - dropout_rate (list): Tasas de dropout [input, hidden1, hidden2, output]
    """
    
    def __init__(self, 
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 128,
                 stage2_out: int = 128,
                 stage3_out: int = 128,
                 dropout_rate: float = 0.2):
        
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels

        # Configuración de capas con parámetros por etapa
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            # Capa entrada -> stage1
            nn.Dropout(dropout_rate),
            nn.Linear(input_channels * time_steps, stage1_out),
            nn.ReLU(),
            
            # stage1 -> stage2
            nn.Dropout(dropout_rate),
            nn.Linear(stage1_out, stage2_out),
            nn.ReLU(),
            
            # stage2 -> stage3
            nn.Dropout(dropout_rate),
            nn.Linear(stage2_out, stage3_out),
            nn.ReLU(),
            
            # stage3 -> salida
            nn.Dropout(dropout_rate),
            nn.Linear(stage3_out, output_channels * time_steps)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplanar entrada (B, C, T) -> (B, C*T)
        x = self.flatten(x)
        
        # Procesar a través de todas las capas
        out = self.layers(x)
        
        # Reformar a la salida deseada (B, output_channels, time_steps)
        return out.view(-1, self.output_channels, self.time_steps)