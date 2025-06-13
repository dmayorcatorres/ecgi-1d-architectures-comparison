import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        return x + self.pe[:, :, :T]


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, D = q.size()
        H = self.n_heads
        q = self.q_proj(q).view(B, T, H, -1).transpose(1, 2)
        k = self.k_proj(k).view(B, T, H, -1).transpose(1, 2)
        v = self.v_proj(v).view(B, T, H, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(context)


class InformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


class Informer1D(nn.Module):
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding(d_model, max_len=time_steps)
        self.encoder = nn.Sequential(*[
            InformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.input_proj(x)              # (B, D, T)
        x = self.pos_enc(x)                 # (B, D, T)
        x = x.permute(0, 2, 1)              # (B, T, D)
        x = self.encoder(x)                 # (B, T, D)
        x = x.permute(0, 2, 1)              # (B, D, T)
        return self.output_proj(x)          # (B, C_out, T)
