import torch
import torch.nn as nn
from src.models.base.ConvSEBlock import ConvSEBlock

class CNN_GRU(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 128,
                 stage2_out: int = 128,
                 stage3_out: int = 128,
                 stage4_out: int = 128,
                 rnn_type: str = "GRU",                # "GRU" or "LSTM"
                 rnn_hidden: int = 16,
                 bidirectional: bool = False,         # Aquí el flag
                 dropout_rate: float = 0.2,
                 se_ratio: int = 16,
                 use_se: bool = True):
        super().__init__()
        # ——— Bloques CNN con SE ———
        self.blocks = nn.ModuleList([
            ConvSEBlock(input_channels, stage1_out, kernel_size=7, se_ratio=se_ratio, use_se=use_se),
            ConvSEBlock(stage1_out,   stage2_out, kernel_size=5, se_ratio=se_ratio, use_se=use_se),
            ConvSEBlock(stage2_out,   stage3_out, kernel_size=3, se_ratio=se_ratio, use_se=use_se),
            ConvSEBlock(stage3_out,   stage4_out, kernel_size=5, se_ratio=se_ratio, use_se=use_se),
        ])
        # ——— RNN branch ———
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_channels,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        # ——— Capa final 1×1 ———
        rnn_out_channels = rnn_hidden * (2 if bidirectional else 1)
        self.fc = nn.Conv1d(
            in_channels=stage4_out + rnn_out_channels,
            out_channels=output_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        cnn = x
        for blk in self.blocks:
            cnn = blk(cnn)  # (B, stage*_out, T)

        # RNN necesita (B, T, C)
        rnn_in = x.permute(0, 2, 1)
        rnn_out, _ = self.rnn(rnn_in)  
        rnn_out = self.dropout(rnn_out)
        rnn_out = rnn_out.permute(0, 2, 1)  # (B, rnn_out_channels, T)

        # Concatenar y proyectar
        combined = torch.cat([cnn, rnn_out], dim=1)
        return self.fc(combined)
