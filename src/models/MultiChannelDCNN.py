import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MultiChannelDCNN(nn.Module):
    """
    MultiChannelDCNN – Modelo MC‑DCNN para clasificación de series temporales multicanal (MIMO).

    Este modelo procesa cada canal (serie univariante) de forma independiente mediante un submodelo,
    que consta de:
      - Dos capas de convolución 1D con padding (para preservar la longitud temporal),
      - Activación ReLU,
      - Batch Normalization,
      - Max Pooling para reducción de resolución temporal,
      - Upsampling para recuperar la longitud original.

    Las salidas de cada submodelo se aplanan y se concatenan para formar un vector de características global,
    que se fusiona mediante una convolución 1×1 (proyección final) para obtener, por cada paso temporal,
    un vector de logits de clase. La salida final se presenta en la forma (B, output_channels, time_steps).

    Parámetros:
      - time_steps (int): Número de pasos temporales (T).
      - input_channels (int): Número de canales de entrada (C).
      - num_classes (int): Número de clases (o salidas).
      - sub_channels (int, opcional): Número de filtros por canal en el submodelo (por defecto 16).
      - kernel_size (int, opcional): Tamaño del kernel para las convoluciones en el submodelo (por defecto 3).
    """
    def __init__(self, 
                 time_steps: int, 
                 input_channels: int, 
                 output_channels: int,
                 sub_channels: int = 16, 
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Submodelo para cada canal (procesa una serie univariante de forma independiente).
        self.submodel = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=sub_channels, kernel_size=kernel_size, 
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=sub_channels, out_channels=sub_channels, kernel_size=kernel_size, 
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),     # Reduce la resolución temporal
            nn.Upsample(scale_factor=2, mode='nearest')  # Recupera la longitud original
        )
        
        # La salida de cada submodelo tendrá la misma longitud temporal que la entrada.
        # Por cada canal se obtienen 'sub_channels' características, por lo que al concatenarlas,
        # la dimensión resultante es: sub_channels * input_channels.
        self.proj = nn.Conv1d(in_channels=sub_channels * input_channels, 
                              out_channels=output_channels, 
                              kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.

        Entrada:
          - x: Tensor de forma (B, input_channels, time_steps), donde:
              B = tamaño del batch,
              input_channels = número de canales (por ejemplo, 163 para ECG),
              time_steps = longitud temporal (por ejemplo, 100).
        
        Retorna:
          - Tensor de salida de forma (B, num_classes, time_steps)
        """
        B, C, T = x.shape  # Se asume que C == input_channels y T == time_steps
        outputs: List[torch.Tensor] = []
        
        # Procesar cada canal de forma independiente.
        for c in range(C):
            # Extraer el canal c: (B, time_steps) y agregar dimensión para Conv1d -> (B, 1, time_steps)
            x_c = x[:, c, :].unsqueeze(1)
            # Aplicar el submodelo: salida (B, sub_channels, time_steps)
            out_c = self.submodel(x_c)
            outputs.append(out_c)
        
        # Concatenar las salidas de todos los canales en la dimensión de canal:
        # Resultado: (B, sub_channels * input_channels, time_steps)
        x_cat = torch.cat(outputs, dim=1)
        
        # Proyección final: fusiona las características para obtener (B, num_classes, time_steps)
        x_out = self.proj(x_cat)
        
        return x_out