import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Codificación posicional senoidal clásica (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (D//2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # pares
        pe[:, 1::2] = torch.cos(position * div_term)  # impares
        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        T = x.size(-1)
        return x + self.pe[:, :, :T]


class Transformer1D(nn.Module):
    """
    Transformer adaptado a series temporales multivariadas:
    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs  # para compatibilidad con ModelFactory
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding(d_model, max_len=time_steps)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.input_proj(x)            # (B, D, T)
        x = self.pos_enc(x)               # (B, D, T)
        x = x.permute(0, 2, 1)            # (B, T, D)
        x = self.encoder(x)               # (B, T, D)
        x = x.permute(0, 2, 1)            # (B, D, T)
        return self.output_proj(x)        # (B, C_out, T)
