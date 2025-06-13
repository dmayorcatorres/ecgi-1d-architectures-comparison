import torch
import torch.nn as nn

class TimeNet1D(nn.Module):
    """
    TimeNet adaptado para series temporales multivariadas (PyTorch).
    Entrada:  (B, C_in, T)
    Salida:   (B, C_out, T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        hidden_size: int = 60,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs  # Compatibilidad con ModelFactory
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada: (B, C_in, T) → (B, T, C_in)
        x = x.permute(0, 2, 1)
        _, h = self.encoder(x)  # h: (num_layers, B, H)

        # Decodificador recibe como entrada la última capa oculta replicada
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, H)
        out, _ = self.decoder(repeated)
        out = out.permute(0, 2, 1)  # (B, H, T)
        return self.output_proj(out)  # (B, C_out, T)
