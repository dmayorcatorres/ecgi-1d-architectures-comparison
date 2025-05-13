import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.base.ConvSEBlock import ConvSEBlock

class CNN_LSTM(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 128,
                 stage2_out: int = 128,
                 stage3_out: int = 128,
                 stage4_out: int = 128,
                 lstm_hidden: int = 16,
                 bidirectional: bool = False,
                 dropout_rate: float = 0.2,
                 se_ratio: int = 16,
                 use_se: bool = True):
        super().__init__()
        # ——— CNN + SE blocks ———
        self.blocks = nn.ModuleList([
            ConvSEBlock(input_channels, stage1_out, 7, se_ratio, use_se),
            ConvSEBlock(stage1_out,   stage2_out,   5, se_ratio, use_se),
            ConvSEBlock(stage2_out,   stage3_out,   3, se_ratio, use_se),
            ConvSEBlock(stage3_out,   stage4_out,   5, se_ratio, use_se),
        ])
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
        self.fc = nn.Conv1d(stage4_out + rnn_ch, output_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        cnn = x
        for blk in self.blocks:
            cnn = blk(cnn)  # (B, stage*_out, T)
        # LSTM expects (B, T, C)
        rnn_in, _ = self.lstm(x.permute(0,2,1))      # (B, T, rnn_ch)
        rnn_out = self.dropout(rnn_in).permute(0,2,1) # (B, rnn_ch, T)
        # concat and project
        return self.fc(torch.cat([cnn, rnn_out], dim=1))