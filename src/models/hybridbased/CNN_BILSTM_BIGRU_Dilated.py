import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_BILSTM_BIGRU_Dilated(nn.Module):
    """
    Regresión de señal EMG desde señal ECG:
    BiGRU + BiLSTM en paralelo con Multilayered Dilated CNN
    y head conv1x1 para reconstruir (B, emg_ch, T).
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,   # aquí emg_channels
                 cnn_channels: int = 128,
                 gru_hidden: int = 64,
                 lstm_hidden: int = 64,
                 dropout: float = 0.5):
        super().__init__()

        # — Multilayered Dilated CNN —
        self.cnn1 = nn.Conv1d(input_channels, cnn_channels,
                               kernel_size=3, padding=1, dilation=1)
        self.cnn2 = nn.Conv1d(cnn_channels, cnn_channels,
                               kernel_size=3, padding=2, dilation=2)
        self.cnn3 = nn.Conv1d(cnn_channels, cnn_channels,
                               kernel_size=3, padding=4, dilation=4)
        self.bn1  = nn.BatchNorm1d(cnn_channels)
        self.bn2  = nn.BatchNorm1d(cnn_channels)
        self.bn3  = nn.BatchNorm1d(cnn_channels)

        # — Ramas Bidireccionales RNN sobre secuencia temporal —
        self.bigru = nn.GRU(input_size=input_channels,
                            hidden_size=gru_hidden,
                            batch_first=True,
                            bidirectional=True)
        self.bilstm = nn.LSTM(input_size=input_channels,
                              hidden_size=lstm_hidden,
                              batch_first=True,
                              bidirectional=True)

        # — Head conv1x1 para reconstrucción temporal —
        total_feats = cnn_channels + 2*gru_hidden + 2*lstm_hidden
        self.head = nn.Conv1d(total_feats, output_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_ecg, T)

        # — Dilated CNN branch → (B, cnn_channels, T) —
        y = self.relu(self.bn1(self.cnn1(x)))
        y = self.relu(self.bn2(self.cnn2(y)))
        y = self.relu(self.bn3(self.cnn3(y)))
        cnn_feat = self.dropout(y)

        # — Preparamos input para RNNs: (B, T, C_ecg) —
        seq = x.permute(0, 2, 1)

        # — BiGRU branch → out: (B, T, 2*gru_hidden) —
        gru_out, _ = self.bigru(seq)
        gru_feat = self.dropout(gru_out.permute(0, 2, 1))  # → (B, 2*gru_hidden, T)

        # — BiLSTM branch → out: (B, T, 2*lstm_hidden) —
        lstm_out, _ = self.bilstm(seq)
        lstm_feat = self.dropout(lstm_out.permute(0, 2, 1))  # → (B, 2*lstm_hidden, T)

        # — Concatenamos las tres ramas por canales → (B, total_feats, T) —
        combined = torch.cat([cnn_feat, gru_feat, lstm_feat], dim=1)

        # — Proyección 1×1 para reconstruir EMG → (B, output_channels, T) —
        out = self.head(combined)
        return out