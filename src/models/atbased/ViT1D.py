import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Codificación posicional senoidal clásica (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # guardamos como (1, T, D) por si quisieras usarla en (B, N, D)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)  -> añadimos PE sobre la dimensión N
        N = x.size(1)
        return x + self.pe[:, :N, :]


class ViT1D(nn.Module):
    """
    Vision Transformer 1D para series temporales multivariadas.
    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)  con T = time_steps exacto (se recorta/acolcha si hace falta).
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
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"
        self.time_steps = int(time_steps)
        self.patch_size = int(patch_size)
        self.output_channels = int(output_channels)
        self.d_model = int(d_model)

        # Número máximo de patches que necesitaremos tras acolchar a múltiplo de P
        self.max_patches = math.ceil(self.time_steps / self.patch_size)

        # Embedding de entrada (conv sobre patches)
        self.input_proj = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True
        )

        # Positional embedding aprendible (tamaño máximo; después hacemos slicing)
        self.pos_embed = nn.Parameter(torch.empty(1, self.max_patches, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.embed_ln = nn.LayerNorm(d_model)
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

        # Cabeza de salida: D -> (patch_size * C_out)
        self.output_proj = nn.Linear(d_model, self.patch_size * self.output_channels)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T_in)
        return: (B, C_out, time_steps)
        """
        B, _, T_in = x.shape

        # Acolcha al siguiente múltiplo de patch_size si hace falta
        pad_t = (-T_in) % self.patch_size
        if pad_t:
            x = F.pad(x, (0, pad_t))  # (B, C, T_in + pad)

        # Embedding por patches
        x = self.input_proj(x)             # (B, D, N)
        x = x.permute(0, 2, 1).contiguous()  # (B, N, D)
        N = x.size(1)

        # PE aprendible (slicing a N)
        x = x + self.pos_embed[:, :N, :]
        x = self.embed_ln(x)
        x = self.dropout(x)

        # Transformer
        x = self.encoder(x)                # (B, N, D)

        # Proyección por patch y reensamblado temporal
        x = self.output_proj(x)                                # (B, N, P*C_out)
        x = x.reshape(B, N, self.output_channels, self.patch_size)  # (B, N, C_out, P)
        x = x.permute(0, 2, 1, 3).contiguous()                 # (B, C_out, N, P)
        x = x.reshape(B, self.output_channels, N * self.patch_size) # (B, C_out, T')

        # Asegurar longitud exacta time_steps
        if x.size(-1) >= self.time_steps:
            x = x[..., :self.time_steps]
        else:
            # Improbable si T_in == time_steps, pero lo dejamos robusto
            missing = self.time_steps - x.size(-1)
            x = F.pad(x, (0, missing))

        return x
