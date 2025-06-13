import torch
import torch.nn as nn
import math

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):  # x: (B, D, T)
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return context

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=25, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, T, D)
        seasonal, trend = self.decomp(x.transpose(1, 2))
        seasonal = seasonal.transpose(1, 2)
        x = self.auto_corr(seasonal)
        x = self.dropout(x)
        return self.norm(x + seasonal), trend

class Autoformer1D(nn.Module):
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.input_proj(x)               # (B, D, T)
        x = x.permute(0, 2, 1)               # (B, T, D)
        for layer in self.encoder_layers:
            x, _ = layer(x)
        x = x.permute(0, 2, 1)               # (B, D, T)
        return self.output_proj(x)          # (B, C_out, T)
