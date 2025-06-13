import torch
import torch.nn as nn
import torch.nn.functional as F

class QRNNLayer(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_size: int,
                 kernel_size: int = 2,
                 pool_type: str = "f",  # "f", "o" o "fo"
                 bias: bool = True):
        super().__init__()
        padding = kernel_size - 1
        self.pool_type = pool_type
        # Z = actualización
        self.conv_z = nn.Conv1d(input_channels, hidden_size,
                                kernel_size, padding=padding, bias=bias)
        # F = olvido
        self.conv_f = nn.Conv1d(input_channels, hidden_size,
                                kernel_size, padding=padding, bias=bias)
        # O = salida, sólo si me pidieron "o" u "fo"
        if pool_type in ("o", "fo"):
            self.conv_o = nn.Conv1d(input_channels, hidden_size,
                                    kernel_size, padding=padding, bias=bias)
        else:
            self.conv_o = None

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        returns y: (B, H, T)
        """
        T = x.size(2)
        Z = torch.tanh(self.conv_z(x))[:, :, :T]
        Fg = torch.sigmoid(self.conv_f(x))[:, :, :T]
        if self.conv_o is not None:
            Og = torch.sigmoid(self.conv_o(x))[:, :, :T]
        else:
            Og = None

        B, H, _ = Z.size()
        h = x.new_zeros(B, H)
        y = []
        for t in range(T):
            zt = Z[:, :, t]
            ft = Fg[:, :, t]
            h = ft * h + (1 - ft) * zt
            if Og is not None:
                # h_tilde = h, luego salida = o_t * h
                h = Og[:, :, t] * h
            y.append(h)
        y = torch.stack(y, dim=2)  # (B, H, T)
        return y



class QRNN(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 hidden_size: int = 64,
                 kernel_size: int = 2,
                 num_layers: int = 1,
                 dropout_rate: float = 0.0,
                 pool_type: str = "f"):

        super().__init__()
        self.time_steps = int(time_steps)

        # ——— Normalizar hiperparámetros a enteros ———
        hidden_size  = int(round(hidden_size))
        kernel_size  = int(round(kernel_size))
        num_layers   = int(round(num_layers))
        assert num_layers > 0,  "num_layers debe ser ≥1"
        assert kernel_size  > 0, "kernel_size debe ser ≥1"
        assert hidden_size  > 0, "hidden_size debe ser ≥1"

        layers = []
        in_ch = input_channels
        for _ in range(num_layers):
            layers.append(
                QRNNLayer(
                    in_ch, hidden_size,
                    kernel_size=kernel_size,
                    pool_type=pool_type
                )
            )
            in_ch = hidden_size

        self.rnn_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, C_in, T)
        y = x
        for layer in self.rnn_layers:
            y = layer(y)
            y = self.dropout(y)
        return self.fc(y)           # (B, C_out, T)
