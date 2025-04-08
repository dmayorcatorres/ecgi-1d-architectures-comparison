import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(ResidualBiLSTMBlock, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        residual = x
        out, _ = self.bilstm(x)  # (B, T, 2 * hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)       # (B, T, input_dim)
        out += residual          # Residual connection
        out = self.layer_norm(out)
        return out

class DeepResidualBiLSTM(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2):
        """
        Parámetros:
          - input_channels: número de canales de entrada (ej. 163)
          - time_steps: número de pasos temporales (ej. 100)
          - output_channels: número de salidas (ej. 502)
          - hidden_dim: tamaño del hidden state para los bloques BiLSTM.
          - num_layers: número de bloques residuales apilados.
        
        La entrada se espera en forma (B, input_channels, T) y la salida será (B, output_channels, T).
        """
        super(DeepResidualBiLSTM, self).__init__()
        self.input_channels = input_channels

        # Bloques residuales BiLSTM
        self.blocks = nn.ModuleList([
            ResidualBiLSTMBlock(input_dim=input_channels, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Capa final para mapeo
        self.fc_final = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,
                                  kernel_size=1, bias=True)

    def forward(self, x):
        """
        x: tensor de entrada con forma (B, input_channels, T)
        Retorna:
            tensor de salida con forma (B, output_channels, T)
        """
        # Cambiamos a (B, T, input_channels) para LSTM
        x = x.permute(0, 2, 1)

        # Aplicamos los bloques residuales
        for block in self.blocks:
            x = block(x)

        # Regresamos a (B, input_channels, T) para capa final
        x = x.permute(0, 2, 1)

        # Mapeo final
        out = self.fc_final(x)  # (B, output_channels, T)
        return out
