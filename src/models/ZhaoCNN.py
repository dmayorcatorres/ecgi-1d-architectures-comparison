import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
      - time_steps (int): Número de pasos temporales de la salida deseada (T). Se usará para replicar la predicción.
      - input_channels (int): Número de canales de la entrada (C) (e.g., 163 para ECG).
      - output_channels (int): Número de clases o canales de salida (e.g., 502 para EMG).
      - stage1_out (int, opcional): Número de filtros de la primera etapa (por defecto 128).
      - kernel_size_stage1 (int, opcional): Tamaño del kernel en la primera etapa (por defecto 5).
      - stage2_out (int, opcional): Número de filtros de la segunda etapa (por defecto 64).
      - kernel_size_stage2 (int, opcional): Tamaño del kernel en la segunda etapa (por defecto 3).
      - stage3_out (Optional[int]): Número de filtros de la tercera etapa (por defecto None, es decir, se usan solo dos etapas).
      - kernel_size_stage3 (int, opcional): Tamaño del kernel en la tercera etapa (por defecto 3).
      - pool_kernel (int, opcional): Tamaño del kernel de max pooling en cada etapa (por defecto 2).

    **Nota:** La capa fully connected se crea dinámicamente en el `forward` usando la forma real después del aplanado.
    """
    def __init__(self, 
                 time_steps: int, 
                 input_channels: int, 
                 output_channels: int,
                 stage1_out: int = 128, 
                 kernel_size_stage1: int = 5,
                 stage2_out: int = 64, 
                 kernel_size_stage2: int = 3,
                 stage3_out: Optional[int] = None, 
                 kernel_size_stage3: int = 3,
                 pool_kernel: int = 2) -> None:
        super().__init__()
        self.time_steps = time_steps  # Este valor se usará para replicar la predicción final
        self.output_channels = output_channels
        self.pool_kernel = pool_kernel

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
        
        # La capa fully connected se creará de forma dinámica en el forward.
        self.fc_dyn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Parámetros:
          - x: Tensor de forma (B, C, T_in) donde T_in es la longitud temporal de entrada.
        
        Retorna:
          - Tensor de forma (B, output_channels, T_out) donde T_out = time_steps (valor fijado).
        """
        B, C, T_in = x.shape

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
        
        # Aplanar la salida: (B, final_channels * T_after)
        B2, final_c, T_after = x.shape
        x = x.view(B2, -1)  # (B2, final_c * T_after)

        # Verificar o crear la capa fully connected de forma dinámica
        in_features = x.shape[1]
        if (self.fc_dyn is None) or (self.fc_dyn.in_features != in_features):
            self.fc_dyn = nn.Linear(in_features, self.output_channels).to(x.device)

        x = self.fc_dyn(x)  # (B2, output_channels)

        # Replicar la predicción a lo largo del eje temporal deseado (time_steps)
        # De esta forma, la salida es (B2, output_channels, time_steps)
        x = x.unsqueeze(2).repeat(1, 1, self.time_steps)
        return x
