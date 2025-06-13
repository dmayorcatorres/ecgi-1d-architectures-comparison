import torch
import torch.nn as nn
from typing import Optional

# ──────────────────────────────────────────────────────────────
#  rnn_gru.py — implementación limpia de RNN_GRU
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class RNN_GRU(nn.Module):
    """Modelo secuencia-a-secuencia basado exclusivamente en GRU."""

    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        gru_hidden: int = 32,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        rnn_feats = gru_hidden * (2 if bidirectional else 1)
        self.proj = nn.Conv1d(rnn_feats, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  →  (B, T, C)
        y, _ = self.gru(x.permute(0, 2, 1))
        # y: (B, T, F)  →  (B, F, T)
        y = y.permute(0, 2, 1)
        return self.proj(y)  # (B, out_C, T)
