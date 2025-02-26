import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        # Utilizamos conv1d de kernel 1 para actuar como Fully Connected en la dimensión de canales
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
        return x * se  # Broadcast a lo largo de T

class LSTM_FCN(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 128, stage2_out: int = 128, stage3_out: int = 128, lstm_out: int = 16):
        """
        Parámetros:
          - input_channels: número de canales de entrada (ej. 163)
          - time_steps: número de pasos temporales (ej. 100)
          - output_channels: número de salidas (ej. 502)
          - stageX_out: número de filtros en cada etapa de la rama CNN.
          - lstm_out: tamaño del hidden state de la rama LSTM.
          
        La entrada se espera en forma (B, input_channels, T) y la salida será (B, output_channels, T).
        """
        super(LSTM_FCN, self).__init__()
        # Rama CNN
        # Bloque 1: conv1d con kernel 8, padding "same"
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=stage1_out, kernel_size=8, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(stage1_out)
        self.se1 = SqueezeExciteBlock(stage1_out, ratio=16)
        
        # Bloque 2: conv1d con kernel 5
        self.conv2 = nn.Conv1d(in_channels=stage1_out, out_channels=stage2_out, kernel_size=5, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(stage2_out)
        self.se2 = SqueezeExciteBlock(stage2_out, ratio=16)
        
        # Bloque 3: conv1d con kernel 3
        self.conv3 = nn.Conv1d(in_channels=stage2_out, out_channels=stage3_out, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(stage3_out)
        
        # Rama LSTM:
        # LSTM espera (B, T, features), por lo que se realiza una única permutación.
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_out, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        
        # Capa final: en lugar de nn.Linear, se utiliza conv1d 1x1 para evitar permutaciones adicionales.
        # La entrada de esta capa es la concatenación de la rama CNN (con stage3_out canales) y la salida del LSTM (lstm_out canales)
        self.fc = nn.Conv1d(in_channels=stage3_out + lstm_out, out_channels=output_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        """
        x: tensor de entrada con forma (B, input_channels, T)
        Retorna:
            tensor de salida con forma (B, output_channels, T)
        """
        # Rama CNN (se trabaja en el formato (B, C, T) sin permutaciones innecesarias)
        cnn = self.conv1(x)         # (B, stage1_out, T)
        cnn = self.bn1(cnn)
        cnn = F.relu(cnn)
        cnn = self.se1(cnn)         # (B, stage1_out, T)
        
        cnn = self.conv2(cnn)       # (B, stage2_out, T)
        cnn = self.bn2(cnn)
        cnn = F.relu(cnn)
        cnn = self.se2(cnn)         # (B, stage2_out, T)
        
        cnn = self.conv3(cnn)       # (B, stage3_out, T)
        cnn = self.bn3(cnn)
        cnn = F.relu(cnn)           # (B, stage3_out, T)
        
        # Rama LSTM:
        # Permutamos x para que tenga forma (B, T, input_channels)
        lstm_in = x.permute(0, 2, 1)  # (B, T, input_channels)
        lstm_out, _ = self.lstm(lstm_in)  # (B, T, lstm_out)
        lstm_out = self.dropout(lstm_out)
        # Volvemos al formato (B, lstm_out, T)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, lstm_out, T)
        
        # Concatenar ambas ramas a lo largo de la dimensión de canales:
        combined = torch.cat([cnn, lstm_out], dim=1)  # (B, stage3_out + lstm_out, T)
        
        # Capa final 1x1 para mapear a output_channels
        out = self.fc(combined)  # (B, output_channels, T)
        return out
