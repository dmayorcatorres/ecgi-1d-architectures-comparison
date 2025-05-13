import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_FCN(nn.Module):
    """
    1D-LSTM-FCN: parallel FCN and LSTM branches fused with a 1×1 conv.
    - FCN branch: 3 × (Conv1d → BN → ReLU)
    - LSTM branch: single LSTM (optional bidirectional) + Dropout
    - Fusion: concat along channels → Conv1d(kernel_size=1) → Output
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
        # ——— FCN branch ———
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, conv1_out, kernel_size=8, padding=4),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(conv2_out, conv3_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv3_out),
            nn.ReLU(inplace=True)
        )
        # ——— LSTM branch ———
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout_rate)
        # ——— Fusion + 1×1 conv ———
        rnn_ch = lstm_hidden * (2 if bidirectional else 1)
        self.fusion_conv = nn.Conv1d(conv3_out + rnn_ch,
                                     output_channels,
                                     kernel_size=1,
                                     bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # — FCN branch —
        f = self.conv1(x)   # (B, conv1_out, T)
        f = self.conv2(f)   # (B, conv2_out, T)
        f = self.conv3(f)   # (B, conv3_out, T)

        # — LSTM branch —
        # permute to (B, T, C)
        r_in = x.permute(0, 2, 1)
        r_out, _ = self.lstm(r_in)             # (B, T, rnn_ch)
        r_out = self.dropout(r_out)
        # back to (B, rnn_ch, T)
        r_out = r_out.permute(0, 2, 1)

        # — Fuse and project —
        combined = torch.cat([f, r_out], dim=1)      # (B, conv3_out + rnn_ch, T)
        return self.fusion_conv(combined)            # (B, output_channels, T)
