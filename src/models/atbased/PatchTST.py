import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Conv1d temporal para obtener parches: (B, C_in, T) → (B, N_patches, D)."""
    def __init__(self, input_channels: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)      # (B, D, N)
        return x.permute(0, 2, 1)    # (B, N, D)


class PatchTST(nn.Module):
    """
    PatchTST minimal (no-CI estricto), corrigiendo la cabeza:
      tokens→tiempo (N→T) y features→canales (D→C_out).
    Salida: (B, C_out, T) sin replicar canales.
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.output_channels = int(output_channels)

        assert self.time_steps >= patch_len, f"T({self.time_steps}) < patch_len({patch_len})"
        self.num_patches = (self.time_steps - patch_len) // stride + 1
        assert self.num_patches >= 1, (
            f"Config produce 0 parches: T={self.time_steps}, P={patch_len}, S={stride}"
        )

        # Embedding por parches
        self.embedding = PatchEmbedding(
            input_channels=input_channels,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )

        # Positional embedding (broadcast)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Encoder Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Cabeza corregida ─────────────────────────────────────
        # 1) Mapear tokens (N_patches) → tiempo (T), para cada feature D
        self.token_to_time = nn.Linear(self.num_patches, self.time_steps)
        # 2) Mapear features D → canales de salida C_out, para cada t
        self.feature_to_channels = nn.Linear(d_model, self.output_channels)
        self.head_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.token_to_time.weight)
        nn.init.zeros_(self.token_to_time.bias)
        nn.init.xavier_uniform_(self.feature_to_channels.weight)
        nn.init.zeros_(self.feature_to_channels.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T) → out: (B, C_out, T)
        """
        # Embedding + PE + Encoder
        x = self.embedding(x)                  # (B, N, D)
        x = x + self.pos_embedding             # (B, N, D)
        x = self.encoder(x)                    # (B, N, D)

        # tokens→tiempo: aplicar Linear(N→T) por cada feature D
        x = x.transpose(1, 2)                  # (B, D, N)
        x = self.token_to_time(x)              # (B, D, T)
        x = self.head_dropout(x)

        # features→canales por cada t
        x = x.transpose(1, 2)                  # (B, T, D)
        x = self.feature_to_channels(x)        # (B, T, C_out)
        x = x.permute(0, 2, 1).contiguous()    # (B, C_out, T)
        return x
