import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

class Dilated2DCNN(nn.Module):
    """
    Dilated2DCNN:
    
    Una arquitectura de red convolucional 2D diseñada para transformar series temporales
    en un escenario MIMO. La red espera una entrada de forma (B, input_channels, time_steps)
    y produce una salida de forma (B, output_channels, time_steps).

    El flujo de procesamiento es el siguiente:
      1. Se expande la entrada a 4D para aplicar convoluciones 2D.
      2. Se aplican dos bloques de convoluciones (dilatadas y con stride) para extraer
         características a lo largo del tiempo, manteniendo la "altura" (canales de entrada).
      3. Se aplica un pooling adaptativo para colapsar la dimensión temporal a 1 (en ancho),
         manteniendo la altura.
      4. La representación resultante se aplana y se proyecta mediante una capa fully connected,
         que mapea a un vector de dimensión (output_channels * time_steps).
      5. Se reestructura la salida a (B, output_channels, time_steps).
    """
    def __init__(
        self, 
        time_steps: int, 
        input_channels: int, 
        output_channels: int, 
        num_filters: int = 32,  # Número de filtros para las capas 2D
        fc_units: int = 256      # Unidades de la capa fully connected
    ):
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels  # Se utiliza como "altura" para las convoluciones 2D
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        # Capas convolucionales 2D con dilatación y stride.
        # Se espera que la entrada a estas capas tenga forma (B, 1, input_channels, time_steps).
        self.dl1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(3, 3),
            dilation=(1, 2),
            padding=(1, 2)
        )
        self.sl1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(1, 4),
            stride=(1, 4),
            padding=(0, 1)
        )
        self.dl2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(3, 3),
            dilation=(1, 2),
            padding=(1, 2)
        )
        self.sl2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=(0, 0)
        )
        
        # Pooling adaptativo: reduce la dimensión del ancho (tiempo) a 1,
        # manteniendo la "altura" igual a input_channels.
        # Salida: (B, num_filters, input_channels, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((input_channels, 1))
        
        # Capa fully connected: mapea la representación a (output_channels * time_steps).
        # La entrada es un vector de tamaño (num_filters * input_channels)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * input_channels, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, output_channels * time_steps)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Parámetros:
          x: Tensor de forma (B, input_channels, time_steps)
             - B: batch size.
             - input_channels: número de canales de entrada (e.g., 163 para ECG).
             - time_steps: longitud temporal (e.g., 272).
        
        Retorna:
          Tensor de forma (B, output_channels, time_steps).
        """
        B, C, T = x.shape  # C debe ser igual a input_channels, T es la longitud temporal
        
        # Expandir la entrada para adaptarla a Conv2d: (B, 1, input_channels, time_steps)
        x = x.unsqueeze(1)
        
        # Aplicar las capas convolucionales 2D con activación ReLU
        out = F.relu(self.dl1(x))
        out = F.relu(self.sl1(out))
        out = F.relu(self.dl2(out))
        out = F.relu(self.sl2(out))
        
        # Pooling adaptativo: reduce la dimensión temporal (ancho) a 1, manteniendo la altura
        # Resultado: (B, num_filters, input_channels, 1)
        out = self.adaptive_pool(out)
        
        # Aplanar la salida: (B, num_filters * input_channels)
        out = out.view(B, -1)
        
        # Proyección final a (B, output_channels * time_steps)
        fc_out = self.fc(out)
        
        # Reestructurar a (B, output_channels, time_steps)
        fc_out = fc_out.view(B, self.output_channels, self.time_steps)
        return fc_out