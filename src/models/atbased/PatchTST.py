import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, input_channels, patch_len, stride, d_model):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_channels, d_model, kernel_size=patch_len, stride=stride)
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        x = self.patch_embed(x)  # (B, D, N_patches)
        return x.permute(0, 2, 1)  # (B, N_patches, D)

class PatchTST(nn.Module):
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
    ):
        super().__init__()

        self.embedding = PatchEmbedding(input_channels, patch_len, stride, d_model)
        num_patches = (time_steps - patch_len) // stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, time_steps)  # Garantiza salida T
        )

        self.time_steps = time_steps
        self.output_channels = output_channels

    def forward(self, x):
        # x: (B, C_in, T)
        x = self.embedding(x)                      # (B, N_patches, D)
        x = x + self.pos_embedding                 # (B, N_patches, D)
        x = self.encoder(x)                        # (B, N_patches, D)
        x = x.mean(dim=1)                          # (B, D)
        x = self.output_proj(x)                    # (B, T)
        x = x.unsqueeze(1).repeat(1, self.output_channels, 1)  # (B, C_out, T)
        return x
