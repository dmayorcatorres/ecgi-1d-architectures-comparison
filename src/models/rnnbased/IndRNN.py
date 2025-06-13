import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 1) Independently RNN (IndRNN)
# ──────────────────────────────────────────────────────────────────────────────
class IndRNNLayer(nn.Module):
    """
    Capa IndRNN: cada neurona tiene W_rec escalar propio.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in  = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.Tensor(hidden_size))
        if bias:
            self.b = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_in, a=math.sqrt(5))
        # recurrent weights positivos para estabilidad
        nn.init.uniform_(self.W_rec, 0, 1)
        if self.b is not None:
            fan_in = self.W_in.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, hx=None, clip_max: float = None):
        """
        x: (B, T, C_in)
        hx: (B, hidden_size)
        clip_max: si no es None, clipea W_rec en [-clip_max, clip_max]
        """
        if clip_max is not None:
            with torch.no_grad():
                self.W_rec.data.clamp_(-clip_max, clip_max)

        B, T, _ = x.size()
        if hx is None:
            hx = x.new_zeros(B, self.hidden_size)
        outputs = []
        h = hx
        for t in range(T):
            xt = x[:, t, :]                        # (B, C_in)
            h = F.linear(xt, self.W_in) + h * self.W_rec
            if self.b is not None:
                h = h + self.b
            h = torch.relu(h)
            outputs.append(h)
        y = torch.stack(outputs, dim=1)            # (B, T, hidden_size)
        return y, h


class IndRNN(nn.Module):
    """
    Red IndRNN multicapa con dropout intermedio y clipping de W_rec.
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout_rate: float = 0.0,
                 clip_max: float = 1.0):
        super().__init__()
        self.num_layers = num_layers
        self.clip_max   = clip_max
        self.time_steps = time_steps

        # Construir capas IndRNN
        layers = []
        in_size = input_channels
        for _ in range(num_layers):
            layers.append(IndRNNLayer(in_size, hidden_size))
            in_size = hidden_size
        self.rnn_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout_rate)

        # Capa de salida 1×1 conv para mantener formato (B, C_out, T)
        self.fc = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C_in, T) → permute a (B, T, C_in)
        y = x.permute(0, 2, 1)
        h = None
        # Propagar por cada capa
        for layer in self.rnn_layers:
            y, h = layer(y, hx=h, clip_max=self.clip_max)
            y = self.dropout(y)  # dropout entre capas

        # y: (B, T, hidden_size) → (B, hidden_size, T)
        y = y.permute(0, 2, 1)
        return self.fc(y)
