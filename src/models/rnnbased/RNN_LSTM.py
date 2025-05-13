import torch
import torch.nn as nn
from typing import Optional

class RNN_LSTM(nn.Module):
    """
    Pure LSTM-based sequence-to-sequence model with optional bidirectionality.
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        lstm_hidden: int = 16,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        rnn_ch = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Conv1d(rnn_ch, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)
        rnn_in = x.permute(0, 2, 1)
        rnn_out, _ = self.lstm(rnn_in)               # (batch, time, rnn_ch)
        rnn_out = rnn_out.permute(0, 2, 1)           # (batch, rnn_ch, time)
        return self.fc(rnn_out)                      # (batch, out_ch, time)
