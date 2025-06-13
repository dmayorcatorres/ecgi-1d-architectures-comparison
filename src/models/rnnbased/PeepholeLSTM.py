import math
import torch
import torch.nn as nn
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Peephole LSTM Cell (Implementación Final Corregida)
# ──────────────────────────────────────────────────────────────────────────────
class PeepholeLSTMCell(nn.Module):
    """Célula LSTM con conexiones peephole."""
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias  # <--- Variable de instancia crítica

        # Parámetros
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.W_co = nn.Parameter(torch.Tensor(hidden_size))

        if self.bias:  # <--- Usar self.bias en lugar de bias
            self.b = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x, hx):
        h_prev, c_prev = hx
        gates = (x @ self.W_x.t()) + (h_prev @ self.W_h.t())
        if self.bias:  # <--- Acceso correcto al atributo
            gates += self.b

        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i + c_prev * self.W_ci)
        f = torch.sigmoid(f + c_prev * self.W_cf)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        o = torch.sigmoid(o + c * self.W_co)
        h = o * torch.tanh(c)
        return h, c


class PeepholeLSTM(nn.Module):
    """LSTM bidireccional con peephole connections implementado correctamente."""
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Crear células por dirección
        self.cells = nn.ModuleList()
        for _ in range(self.num_directions):
            layer_cells = nn.ModuleList()
            in_size = input_channels
            for _ in range(num_layers):
                layer_cells.append(PeepholeLSTMCell(in_size, hidden_size))
                in_size = hidden_size * self.num_directions  # Ajuste para capas profundas
            self.cells.append(layer_cells)

        # Capa de proyección final
        self.fc = nn.Conv1d(
            in_channels=hidden_size * self.num_directions,
            out_channels=output_channels,
            kernel_size=1
        )

    def forward(self, x):
        # x: (B, C, T) → (T, B, C)
        x = x.permute(2, 0, 1)
        seq_len, batch_size, _ = x.shape

        outputs = []
        for direction in range(self.num_directions):
            x_dir = x.flip(0) if direction == 1 else x
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

            direction_outputs = []
            for t in range(seq_len):
                inp = x_dir[t]
                for layer in range(self.num_layers):
                    # Corrección clave: paréntesis cerrado
                    h_layer, c_layer = self.cells[direction][layer](inp, (h[layer], c[layer]))
                    h[layer] = h_layer
                    c[layer] = c_layer
                    inp = h_layer
                    # Concatenar solo si hay más de una capa y es bidireccional
                    if layer < self.num_layers - 1 and self.bidirectional:
                        inp = torch.cat([inp, h[layer]], dim=1)
                direction_outputs.append(h_layer)

            if direction == 1:
                direction_outputs = direction_outputs[::-1]
            outputs.append(torch.stack(direction_outputs, dim=2))

        y = torch.cat(outputs, dim=1) if self.bidirectional else outputs[0]
        return self.fc(y)
