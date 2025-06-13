import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_FCN(nn.Module):
    """
    1D-LSTM-FCN: ramas FCN y LSTM en paralelo, fusionadas con una convolución 1×1.
    - Ramo FCN: 3 × (Conv1d → BatchNorm1d → ReLU), con padding="same" para conservar longitud T.
    - Ramo LSTM: un LSTM (opcional bidireccional) + Dropout.
    - Fusión: concat en canales → Conv1d(kernel_size=1) → salida.
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 conv1_out: int = 128,
                 conv2_out: int = 256,
                 conv3_out: int = 128,
                 lstm_hidden: int = 128,
                 bidirectional: bool = False,
                 dropout_rate: float = 0.8):
        super().__init__()
        # ——— Ramo FCN ———
        # Usamos padding="same" para asegurarnos de que cada conv1d devuelva longitud T en la dimensión temporal.
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, conv1_out, kernel_size=8, padding="same"),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out,      conv2_out, kernel_size=5, padding="same"),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(conv2_out,      conv3_out, kernel_size=3, padding="same"),
            nn.BatchNorm1d(conv3_out),
            nn.ReLU(inplace=True)
        )

        # ——— Ramo LSTM ———
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout_rate)

        # ——— Capa de fusión final (1×1 conv) ———
        rnn_ch = lstm_hidden * (2 if bidirectional else 1)
        self.fusion_conv = nn.Conv1d(conv3_out + rnn_ch,
                                     output_channels,
                                     kernel_size=1,
                                     bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)

        # — FCN branch —
        f = self.conv1(x)   # (B, conv1_out, T)
        f = self.conv2(f)   # (B, conv2_out, T)
        f = self.conv3(f)   # (B, conv3_out, T)

        # — LSTM branch —
        r_in = x.permute(0, 2, 1)          # (B, T, C_in)
        r_out, _ = self.lstm(r_in)         # (B, T, rnn_ch)
        r_out = self.dropout(r_out)
        r_out = r_out.permute(0, 2, 1)      # (B, rnn_ch, T)

        # — Fusionamos ambas ramas (tienen ambas dimensión temporal T) —
        combined = torch.cat([f, r_out], dim=1)  # (B, conv3_out + rnn_ch, T)
        return self.fusion_conv(combined)        # (B, output_channels, T)
