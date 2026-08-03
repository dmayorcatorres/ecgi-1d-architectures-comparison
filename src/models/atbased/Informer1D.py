import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding para tensores en formato (B, D, T).
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # (T, D/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (T, D/2)

        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        T = x.size(-1)
        pe = self.pe[:, :, :T].to(dtype=x.dtype, device=x.device)
        return x + pe


class ProbSparseAttention(nn.Module):
    """
    Ojo: esto implementa atención *densa* multi-head.
    Si quieres ProbSparse real, hay que submuestrear keys/queries (top-k).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} debe ser divisible por num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, T, D)
        B, T, D = q.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(q).reshape(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_proj(k).reshape(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        v = self.v_proj(v).reshape(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B, H, T, hd)
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)  # (B, T, D)

        return self.out_proj(context)


class InformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop_res1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.drop_res1(self.attn(x, x, x))
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
        **kwargs,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} debe ser divisible por num_heads={num_heads}")

        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding(d_model, max_len=time_steps)

        self.layers = nn.ModuleList(
            [InformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.input_proj(x)          # (B, D, T)
        x = self.pos_enc(x)             # (B, D, T)
        x = x.permute(0, 2, 1)          # (B, T, D)

        for layer in self.layers:
            x = layer(x)                # (B, T, D)

        x = x.permute(0, 2, 1)          # (B, D, T)
        return self.output_proj(x)      # (B, C_out, T)
