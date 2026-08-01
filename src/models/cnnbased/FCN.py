import torch
import torch.nn as nn
from typing import Sequence, Union
from models.base.ResidualBlock import ResBlock3x


# =========================
# MLP (salida: B, C_out, T)
# =========================
class MLP(nn.Module):
    """
    MLP para reconstrucción de series temporales.

    Parámetros:
      - time_steps (int): Pasos temporales de salida (y esperados de entrada).
      - input_channels (int): Canales de entrada.
      - output_channels (int): Canales de salida.
      - stage1_out (int): Neuronas capa oculta 1 (default=128).
      - stage2_out (int): Neuronas capa oculta 2 (default=128).
      - stage3_out (int): Neuronas capa oculta 3 (default=128).
      - dropout_rate (float | Sequence[float]): tasa única o lista de 4 tasas
        [in, h1, h2, out]. Si se da un float, se replica a las 4 posiciones.
    """

    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        stage1_out: int = 128,
        stage2_out: int = 128,
        stage3_out: int = 128,
        dropout_rate: Union[float, Sequence[float]] = 0.2,
    ):
        super().__init__()
        self.time_steps = int(time_steps)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)

        d0, d1, d2, d3 = self._parse_dropout(dropout_rate)

        in_features = self.input_channels * self.time_steps
        out_features = self.output_channels * self.time_steps

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Dropout(d0),
            nn.Linear(in_features, stage1_out),
            nn.ReLU(inplace=True),

            nn.Dropout(d1),
            nn.Linear(stage1_out, stage2_out),
            nn.ReLU(inplace=True),

            nn.Dropout(d2),
            nn.Linear(stage2_out, stage3_out),
            nn.ReLU(inplace=True),

            nn.Dropout(d3),
            nn.Linear(stage3_out, out_features),
        )

        self._init_weights()

    def _parse_dropout(self, p: Union[float, Sequence[float]]):
        if isinstance(p, (float, int)):
            return float(p), float(p), float(p), float(p)
        p = list(p)
        assert len(p) == 4, "dropout_rate debe tener 4 valores [in, h1, h2, out]"
        return tuple(float(x) for x in p)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        B, C, T = x.shape
        assert C == self.input_channels, f"Se esperaban {self.input_channels} canales, llega {C}"
        assert T == self.time_steps, f"Se esperaban T={self.time_steps}, llega {T}"

        x = self.flatten(x)                  # (B, C_in*T)
        out = self.layers(x)                 # (B, C_out*T)
        out = out.reshape(B, self.output_channels, self.time_steps)
        return out


# ===========================================
# FCN (salida: B, C_out, T) sin AdaptivePool
# ===========================================
class FCN(nn.Module):
    """
    Fully Convolutional Network para reconstrucción de series temporales.

    Parámetros:
      - time_steps (int): longitud temporal de salida (y esperada de entrada).
      - input_channels (int): canales de entrada.
      - output_channels (int): canales de salida.
      - stage1_out (int): nº de filtros en los bloques residuales (default=64).
      - dropout_rate (float): dropout antes de la capa fully connected final.
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        stage1_out: int = 64,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.time_steps = int(time_steps)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)

        self.block1 = ResBlock3x(self.input_channels, stage1_out)
        self.block2 = ResBlock3x(stage1_out, stage1_out)
        self.block3 = ResBlock3x(stage1_out, stage1_out)

        # Reducción temporal estable para torch.compile (evita Adaptive* pooling):
        # Usamos max sobre el eje temporal en forward.
        self.pre_fc_dropout = nn.Dropout(float(dropout_rate))
        self.fc = nn.Linear(stage1_out, self.output_channels * self.time_steps)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=0.01)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        B, C, T = x.shape
        assert C == self.input_channels, f"Se esperaban {self.input_channels} canales, llega {C}"
        assert T == self.time_steps, f"Se esperaban T={self.time_steps}, llega {T}"

        out = self.block1(x)     # (B, stage1_out, T)
        out = self.block2(out)   # (B, stage1_out, T)
        out = self.block3(out)   # (B, stage1_out, T)

        # Reducción temporal robusta (sin AdaptiveMaxPool1d):
        out = torch.amax(out, dim=-1)  # (B, stage1_out)

        out = self.pre_fc_dropout(out)
        out = self.fc(out)             # (B, C_out*T)
        out = out.reshape(B, self.output_channels, self.time_steps)
        return out
