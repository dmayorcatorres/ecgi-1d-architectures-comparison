import math
import torch
import torch.nn as nn
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Peephole LSTM Cell + Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class PeepholeLSTMCell(nn.Module):
    """
    LSTM Cell with peephole connections: each gate receives además la salida
    de la celda (c_{t-1}) como input lineal.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        # Pesos de puerta de entrada, olvido, celda candidata y salida
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        # Peephole: un parámetro por gate (i,f,o) conectado a c_{t-1}
        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.W_co = nn.Parameter(torch.Tensor(hidden_size))

        if bias:
            self.b = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x, hx):
        """
        x: (B, input_size)
        hx: tuple(h_{t-1}, c_{t-1}) each of shape (B, hidden_size)
        """
        h_prev, c_prev = hx
        # cálculos lineales
        gates = (x @ self.W_x.t()) + (h_prev @ self.W_h.t())
        if self.bias:
            gates = gates + self.b

        # split
        i, f, g, o = gates.chunk(4, dim=1)
        # sumar peephole
        i = torch.sigmoid(i + c_prev * self.W_ci)
        f = torch.sigmoid(f + c_prev * self.W_cf)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        o = torch.sigmoid(o + c * self.W_co)
        h = o * torch.tanh(c)
        return h, c


class PeepholeLSTM(nn.Module):
    """
    Wrapper para procesar secuencia completa con PeepholeLSTMCell.
    input: (B, C, T) → output: (B, hidden_size, T)
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # crear célula por capa
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_channels if layer == 0 else hidden_size * self.num_directions
            self.cells.append(PeepholeLSTMCell(in_size, hidden_size))

        # proyección final
        self.fc = nn.Conv1d(hidden_size * self.num_directions, output_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T) → permutar para (T, B, C)
        x = x.permute(2, 0, 1)
        seq_len, B, _ = x.size()
        # estados iniciales
        h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        c = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[t]
            for l, cell in enumerate(self.cells):
                h_l, c_l = cell(inp, (h[l], c[l]))
                h[l], c[l] = h_l, c_l
                inp = h_l
            outputs.append(h_l)

        # stack y proyección
        y = torch.stack(outputs, dim=2)  # (B, hidden_size, T)
        return self.fc(y)