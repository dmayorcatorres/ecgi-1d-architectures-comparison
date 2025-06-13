# ──────────────────────────────────────────────────────────────────────────────
# 3) Echo State Network (ESN)
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESN(nn.Module):
    """
    Echo State Network:
      - W_in, W_res fijos y aleatorios
      - sólo W_out se entrena
      - ridge_alpha para regularización Ridge en W_out (por implementar en entrenamiento)
    """
    def __init__(self,
                 time_steps, input_channels, output_channels,
                 reservoir_size: int = 500,
                 spectral_radius: float = 0.9,
                 sparsity: float = 0.1,
                 bias: bool = True,
                 ridge_alpha: float = 1e-3):
        super().__init__()
        self.res_size = reservoir_size
        self.ridge_alpha = ridge_alpha  # ahora aceptado

        # Entrada → reserva
        self.W_in  = nn.Parameter(
            torch.empty(reservoir_size, input_channels).normal_(0, 1),
            requires_grad=False
        )

        # Matriz recurrente dispersa
        W = torch.randn(reservoir_size, reservoir_size)
        mask = torch.rand(reservoir_size, reservoir_size) < sparsity
        W *= mask.float()
        # ajustar radio espectral
        eigs = torch.linalg.eigvals(W)
        radius = eigs.abs().max()
        W *= (spectral_radius / radius)
        self.W_res = nn.Parameter(W, requires_grad=False)

        # lectura (entrenable)
        self.W_out = nn.Parameter(
            torch.empty(output_channels, reservoir_size).uniform_(-0.1, 0.1)
        )
        self.b     = nn.Parameter(torch.zeros(output_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        """
        B, _, T = x.size()
        h = x.new_zeros(B, self.res_size)
        states = []
        for t in range(T):
            u = x[:, :, t]  # (B, C_in)
            h = torch.tanh(
                F.linear(u, self.W_in) +
                h @ self.W_res.t()
            )
            states.append(h)
        S = torch.stack(states, dim=2)           # (B, res_size, T)

        # proyectar con W_out
        # F.linear quiere (B*T, res_size), pero con permisos funciona:
        Y = F.linear(S.permute(0, 2, 1), self.W_out, self.b)  # (B, T, C_out)
        return Y.permute(0, 2, 1)                   # (B, C_out, T)
