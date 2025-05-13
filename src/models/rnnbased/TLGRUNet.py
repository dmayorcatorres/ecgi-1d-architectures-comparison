import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["TLGRUCell", "TLGRUNet"]

# -----------------------------------------------------------------------------
#  Célula TLGRU: incorpora decaimiento según *delta‑t* y *delta‑l* (intervalo
#  temporal y de localización).  Si no se proporcionan, usa ceros -> gamma = 1.
# -----------------------------------------------------------------------------
class TLGRUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Puertas estándar GRU (lineales combinadas para eficiencia)
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

        # Parámetros aprendibles para ponderar deltas
        self.w_time = nn.Parameter(torch.randn(hidden_size))  # α_t
        self.w_loc  = nn.Parameter(torch.randn(hidden_size))  # α_l

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        delta_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Un paso de tiempo.

        x_t      : (B, input_size)
        h_prev   : (B, hidden_size)
        delta_t  : (B, 1) ó (B,)  – intervalo temporal desde la última obs.
        delta_l  : (B, 1) ó (B,)  – intervalo de localización
        """
        if delta_t is None:
            delta_t = torch.zeros(x_t.size(0), device=x_t.device)
        if delta_l is None:
            delta_l = torch.zeros(x_t.size(0), device=x_t.device)

        # → (B, hidden_size) tras broadcast
        gamma = torch.exp(
            -F.relu(
                self.w_time * delta_t.unsqueeze(-1)
                + self.w_loc * delta_l.unsqueeze(-1)
            )
        )
        h_prev_hat = h_prev * gamma  # decaimiento

        gates = self.x2h(x_t) + self.h2h(h_prev_hat)
        r, z, n = gates.chunk(3, dim=-1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(r * n)

        h_new = (1 - z) * n + z * h_prev_hat
        return h_new


# -----------------------------------------------------------------------------
#                          Red TLGRU completa
# -----------------------------------------------------------------------------
class TLGRUNet(nn.Module):
    """TLGRU secuencial → proyección 1×1 para salida paso‑a‑paso.

    Compatible con el resto del pipeline: firma (time_steps, input_channels,
    output_channels, ...). Si no se proporcionan *delta_t* / *delta_l* en
    `forward`, asume ceros y se comporta como un GRU clásico.
    """

    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.cells = nn.ModuleList(
            [
                TLGRUCell(
                    input_size=input_channels if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for i in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_channels)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        delta_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, C_in, T)  – Devuelve (B, C_out, T)"""
        B, C, T = x.shape
        h = [torch.zeros(B, cell.hidden_size, device=x.device) for cell in self.cells]
        outs = []
        # reordenar a (T, B, C)
        x_seq = x.permute(2, 0, 1)
        if delta_t is None:
            delta_t = torch.zeros(T, B, device=x.device)
        if delta_l is None:
            delta_l = torch.zeros(T, B, device=x.device)

        for t in range(T):
            inp = x_seq[t]
            dt = delta_t[t]
            dl = delta_l[t]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l], dt, dl)
                inp = h[l]
            outs.append(self.head(self.dropout(inp)))

        y = torch.stack(outs, dim=2)  # (B, C_out, T)
        return y


if __name__ == "__main__":
    B, C_in, T = 4, 3, 50
    x = torch.randn(B, C_in, T)
    model = TLGRUNet(T, C_in, 7, hidden_size=32, num_layers=2)
    y = model(x)
    print(y.shape)
