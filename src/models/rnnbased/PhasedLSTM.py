import math
import torch
import torch.nn as nn
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Phased LSTM Cell + Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class PhasedLSTMCell(nn.Module):
    """
    Phased LSTM Cell (Neil et al., 2016):
      - incorpora un 'time gate' k_t definido por periodo, fase y ratio_on.
      - requiere timestamps en entrada.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 period_init: float = 1.0,
                 r_on_init: float = 0.05,
                 bias: bool = True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # pesos estándar LSTM (sin peephole)
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b   = nn.Parameter(torch.Tensor(4 * hidden_size)) if bias else None

        # parámetros del time gate
        self.period = nn.Parameter(torch.Tensor(hidden_size))
        self.phase  = nn.Parameter(torch.Tensor(hidden_size))
        self.r_on   = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters(period_init, r_on_init)

    def reset_parameters(self, period_init, r_on_init):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in [self.W_x, self.W_h]:
            nn.init.uniform_(p, -stdv, stdv)
        if self.b is not None:
            nn.init.uniform_(self.b, -stdv, stdv)
        # inicializar periodos y fases aleatoriamente
        nn.init.uniform_(self.period, 0.5 * period_init, 1.5 * period_init)
        nn.init.uniform_(self.phase, 0.0, period_init)
        nn.init.constant_(self.r_on, r_on_init)

    def forward(self, x, hx, timestamp):
        """
        x:          (B, input_size)
        hx:         (h_prev, c_prev), cada uno (B, hidden_size)
        timestamp:  (B,) tiempo continuo para cada muestra
        """
        h_prev, c_prev = hx
        # standard LSTM gates
        gates = x @ self.W_x.t() + h_prev @ self.W_h.t()
        if self.b is not None:
            gates = gates + self.b
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        o = torch.sigmoid(o)

        # compute phase of each neuron
        # ϕ = ((t - phase) mod period) / period
        phi = ((timestamp.unsqueeze(1) - self.phase) % self.period) / self.period
        # time gate k_t
        k = torch.where(
            phi < self.r_on / 2,
            2 * phi / self.r_on,
            torch.where(
                phi < self.r_on,
                2 - 2 * phi / self.r_on,
                torch.zeros_like(phi)
            )
        )
        # output update only when gate open
        h = k * (o * torch.tanh(c)) + (1 - k) * h_prev
        return h, c


class PhasedLSTM(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 period_init: float = 1.0,    # Nuevo parámetro
                 r_on_init: float = 0.05):    # Nuevo parámetro
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            PhasedLSTMCell(
                input_channels if l == 0 else hidden_size,
                hidden_size,
                period_init=period_init,  # Pasa los hiperparámetros a la célula
                r_on_init=r_on_init
            )
            for l in range(num_layers)
        ])
        self.fc = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x, times):
        # x:     (B, C, T)
        # times: (B, T) timestamps in same units que period
        x = x.permute(2, 0, 1)      # → (T, B, C)
        times = times.permute(1, 0) # → (T, B)
        T, B, _ = x.size()

        h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        c = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        outputs = []

        for t in range(T):
            inp = x[t]
            ts  = times[t]
            for l, cell in enumerate(self.cells):
                h_l, c_l = cell(inp, (h[l], c[l]), ts)
                h[l], c[l] = h_l, c_l
                inp = h_l
            outputs.append(h_l)

        y = torch.stack(outputs, dim=2)  # (B, hidden_size, T)
        return self.fc(y)