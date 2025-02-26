import torch
import torch.nn as nn
import torch.nn.functional as F

class MCCNN(nn.Module):
    """
    MCCNN_MIMO – Modelo MC‑CNN para problemas MIMO en series temporales.

    Este modelo procesa la entrada de forma (B, canales, tiempo) mediante tres etapas de
    convolución 1D. En cada etapa se aplica:
      - Convolución 1D con padding para preservar la longitud temporal,
      - Batch Normalization,
      - Activación ReLU,
      - Max Pooling para extraer características relevantes y reducir la resolución temporal.
      
    Tras las tres etapas, la representación resultante (con resolución temporal reducida)
    se aplana y se pasa por una capa fully connected que fusiona las características, 
    produciendo un vector de dimensión (output_channels * T_reduced). Este vector se
    reestructura a (B, output_channels, T_reduced) y se aplica un upsampling para restaurar
    la longitud temporal original, generando una salida de forma (B, output_channels, tiempo).

    Parámetros:
      - time_steps (int): Longitud temporal de la entrada (T).
      - input_channels (int): Número de canales de la entrada.
      - output_channels (int): Número de canales de salida.
      - stage1_out (int, opcional): Número de filtros en la primera etapa (por defecto 128).
      - kernel_size_stage1 (int, opcional): Tamaño del kernel en la primera etapa (por defecto 5).
      - pool1 (int, opcional): Tamaño del max pooling en la primera etapa (por defecto 2).
      - stage2_out (int, opcional): Número de filtros en la segunda etapa (por defecto 64).
      - kernel_size_stage2 (int, opcional): Tamaño del kernel en la segunda etapa (por defecto 3).
      - pool2 (int, opcional): Tamaño del max pooling en la segunda etapa (por defecto 2).
      - stage3_out (int, opcional): Número de filtros en la tercera etapa (por defecto 64).
      - kernel_size_stage3 (int, opcional): Tamaño del kernel en la tercera etapa (por defecto 3).
      - pool3 (int, opcional): Tamaño del max pooling en la tercera etapa (por defecto 2).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 128, kernel_size_stage1: int = 5, pool1: int = 2,
                 stage2_out: int = 64, kernel_size_stage2: int = 3, pool2: int = 2,
                 stage3_out: int = 64, kernel_size_stage3: int = 3, pool3: int = 2) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels
        
        # Bloque 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=stage1_out,
                      kernel_size=kernel_size_stage1,
                      padding=kernel_size_stage1 // 2),
            nn.BatchNorm1d(stage1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool1)
        )
        
        # Bloque 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=stage1_out,
                      out_channels=stage2_out,
                      kernel_size=kernel_size_stage2,
                      padding=kernel_size_stage2 // 2),
            nn.BatchNorm1d(stage2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool2)
        )
        
        # Bloque 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=stage2_out,
                      out_channels=stage3_out,
                      kernel_size=kernel_size_stage3,
                      padding=kernel_size_stage3 // 2),
            nn.BatchNorm1d(stage3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool3)
        )
        
        # Calcular la longitud temporal reducida tras los tres max pooling
        self.reduced_T = time_steps // (pool1 * pool2 * pool3)
        
        # Capa fully connected para fusionar las características:
        # La entrada es de tamaño (B, stage3_out * reduced_T)
        # La salida es de tamaño (B, output_channels * reduced_T)
        self.fc = nn.Sequential(
            nn.Linear(stage3_out * self.reduced_T, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_channels * self.reduced_T)
        )
        
        # Upsample para restaurar la longitud temporal original:
        # Se utiliza interpolación lineal para expandir de reduced_T a time_steps.
        self.upsample = nn.Upsample(scale_factor=(time_steps / self.reduced_T), mode='linear', align_corners=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante.
        
        Parámetros:
          - x: Tensor de entrada de forma (B, canales, tiempo)
        
        Retorna:
          - Tensor de salida de forma (B, output_channels, tiempo)
        """
        # x: (B, input_channels, T)
        x = self.conv1(x)  # (B, stage1_out, T1)
        x = self.conv2(x)  # (B, stage2_out, T2)
        x = self.conv3(x)  # (B, stage3_out, T_reduced)  con T_reduced = time_steps/(pool1*pool2*pool3)
        
        B, C, T_reduced = x.shape
        # Aplanar la salida para la capa fully connected: (B, stage3_out * T_reduced)
        x_flat = x.view(B, -1)
        # Capa fully connected: (B, output_channels * T_reduced)
        x_fc = self.fc(x_flat)
        # Reshape a (B, output_channels, T_reduced)
        x_fc = x_fc.view(B, self.output_channels, T_reduced)
        # Upsample la dimensión temporal para recuperar time_steps
        x_out = self.upsample(x_fc)
        return x_out