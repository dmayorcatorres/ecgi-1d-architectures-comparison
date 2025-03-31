import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloque Squeeze-and-Excite (opcional, se usa para recalibrar canales)
class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels // ratio, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, T)
        se = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)
        se = self.fc1(se)                # (B, C//ratio, 1)
        se = self.relu(se)
        se = self.fc2(se)                # (B, C, 1)
        se = self.sigmoid(se)
        return x * se  # Se aplica broadcasting sobre la dimensión temporal

# Arquitectura 1D CNN-GRU
class CNN_GRU(nn.Module):
    def __init__(self, 
                 time_steps: int, 
                 input_channels: int, 
                 output_channels: int,
                 stage1_out: int = 128, 
                 kernel_size_stage1: int = 8,
                 stage2_out: int = 128, 
                 kernel_size_stage2: int = 5,
                 stage3_out: int = 128, 
                 kernel_size_stage3: int = 3,
                 gru_out: int = 16,
                 use_se: bool = True):
        """
        Parámetros:
          - time_steps (int): Número de pasos temporales de la entrada (T).
          - input_channels (int): Número de canales de la entrada (C) (por ejemplo, 163 para ECG).
          - output_channels (int): Número de clases o canales de salida (por ejemplo, 502 para EMG).
          - stageX_out: Número de filtros en cada etapa de la rama CNN.
          - kernel_size_stageX: Tamaño del kernel en cada etapa de la rama CNN.
          - gru_out (int): Tamaño del estado oculto de la capa GRU.
          - use_se (bool): Si True, se aplican bloques Squeeze-and-Excite en las etapas CNN.
          
        Se asume que la entrada tiene forma (B, input_channels, T) y la salida tendrá forma (B, output_channels, T).
        """
        super(CNN_GRU, self).__init__()
        
        # Rama CNN
        self.conv1 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=stage1_out, 
                               kernel_size=kernel_size_stage1,
                               padding=kernel_size_stage1 // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(stage1_out)
        self.se1 = SqueezeExciteBlock(stage1_out) if use_se else nn.Identity()
        
        self.conv2 = nn.Conv1d(in_channels=stage1_out, 
                               out_channels=stage2_out, 
                               kernel_size=kernel_size_stage2,
                               padding=kernel_size_stage2 // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(stage2_out)
        self.se2 = SqueezeExciteBlock(stage2_out) if use_se else nn.Identity()
        
        self.conv3 = nn.Conv1d(in_channels=stage2_out, 
                               out_channels=stage3_out, 
                               kernel_size=kernel_size_stage3,
                               padding=kernel_size_stage3 // 2,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(stage3_out)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Rama GRU
        # GRU espera la entrada en forma (B, T, features) → se permuta la entrada
        self.gru = nn.GRU(input_size=input_channels, hidden_size=gru_out, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        
        # Capa final: Se concatena la salida de la rama CNN y la rama GRU y se mapea a output_channels.
        # Para evitar permutaciones adicionales se usa Conv1d 1x1.
        # La entrada de esta capa tendrá (stage3_out + gru_out) canales.
        self.fc = nn.Conv1d(in_channels=stage3_out + gru_out, out_channels=output_channels, kernel_size=1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_channels, T)
        
        # Rama CNN:
        cnn = self.conv1(x)         # (B, stage1_out, T)
        cnn = self.bn1(cnn)
        cnn = self.relu(cnn)
        cnn = self.se1(cnn)         # (B, stage1_out, T)
        
        cnn = self.conv2(cnn)       # (B, stage2_out, T)
        cnn = self.bn2(cnn)
        cnn = self.relu(cnn)
        cnn = self.se2(cnn)         # (B, stage2_out, T)
        
        cnn = self.conv3(cnn)       # (B, stage3_out, T)
        cnn = self.bn3(cnn)
        cnn = self.relu(cnn)        # (B, stage3_out, T)
        
        # Rama GRU:
        # Permutamos para que la entrada tenga forma (B, T, input_channels)
        gru_in = x.permute(0, 2, 1)   # (B, T, input_channels)
        gru_out, _ = self.gru(gru_in)  # (B, T, gru_out)
        gru_out = self.dropout(gru_out)
        gru_out = gru_out.permute(0, 2, 1)  # (B, gru_out, T)
        
        # Concatenamos las ramas a lo largo del eje de canales:
        combined = torch.cat([cnn, gru_out], dim=1)  # (B, stage3_out + gru_out, T)
        
        # Capa final 1x1 para mapear a output_channels:
        out = self.fc(combined)  # (B, output_channels, T)
        return out
