import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Codificación posicional senoidal clásica (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        T = x.size(-1)
        return x + self.pe[:, :, :T]


class ViT1D(nn.Module):
    """
    Vision Transformer 1D para series temporales multivariadas:
    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        patch_size: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        assert time_steps % patch_size == 0, "El número de pasos temporales debe ser divisible por el tamaño de patch"
        self.patch_size = patch_size
        self.num_patches = time_steps // patch_size
        self.d_model = d_model
        self.time_steps = time_steps
        self.output_channels = output_channels

        # Embedding de entrada (conv sobre patch)
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embedding aprendible
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decodificación a patch_size * C_out → luego se reensambla en tiempo
        self.output_proj = nn.Linear(d_model, patch_size * output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.input_proj(x)      # (B, D, N_patches)
        x = x.permute(0, 2, 1)      # (B, N_patches, D)
        x = x + self.pos_embed      # (B, N_patches, D)
        x = self.dropout(x)
        x = self.encoder(x)         # (B, N_patches, D)
        x = self.output_proj(x)     # (B, N_patches, patch_size * C_out)

        # Reensamblar patches → (B, C_out, T)
        B, N, _ = x.shape
        x = x.view(B, N, self.output_channels, self.patch_size)         # (B, N, C_out, P)
        x = x.permute(0, 2, 1, 3).reshape(B, self.output_channels, -1)  # (B, C_out, T)

        return x
