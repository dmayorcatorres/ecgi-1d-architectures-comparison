# models/cnnbased/ResNetCNN.py
import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    """Bloque básico ResNet para 1D: conv3x1 → BN → ReLU → conv3x1 → BN + atajo → ReLU.
    Conserva la longitud temporal T cuando stride=1 y padding=1."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        # Nota: usamos bias=False porque hay BatchNorm
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            # Atajo 1x1 para igualar dimensiones de canal o stride (si se usara !=1)
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion),
            )

        # Inicialización Kaiming recomendada en ResNet
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample[0].weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNetCNN(nn.Module):
    """
    ResNet1D “clásica” para series temporales:
      stem: Conv1d(k=kernel_size_in, padding=same) + BN + ReLU
      body: num_blocks × BasicBlock1D (sin cambiar T)
      head: Conv1d(1×1) → output_channels (manteniendo T)

    Parámetros (NO modificados):
      - time_steps (int): longitud temporal T de la entrada/salida
      - input_channels (int): canales de entrada C_in
      - output_channels (int): canales de salida C_out
      - num_blocks (int, opcional): nº de bloques residuales (por defecto 1)
      - stage1_out (int, opcional): nº de canales intermedios (por defecto 64)
      - kernel_size_in (int, opcional): kernel de la conv inicial (por defecto 7)
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 num_blocks: int = 1,
                 stage1_out: int = 64,
                 kernel_size_in: int = 7) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.output_channels = output_channels

        # ── Stem ───────────────────────────────────────────────────────────
        padding = kernel_size_in // 2
        self.conv_in = nn.Conv1d(input_channels, stage1_out,
                                 kernel_size=kernel_size_in,
                                 padding=padding, bias=False)
        self.bn_in   = nn.BatchNorm1d(stage1_out)
        self.relu    = nn.ReLU(inplace=True)

        # ── Cuerpo ResNet (T se preserva: stride=1 y padding=1) ───────────
        self.inplanes = stage1_out
        self.layers = self._make_layer(BasicBlock1D, planes=stage1_out,
                                       blocks=num_blocks, stride=1)

        # ── Head temporal: proyección a C_out manteniendo T ───────────────
        self.head = nn.Conv1d(stage1_out, output_channels, kernel_size=1, bias=True)

        # Inicialización del stem
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity="relu")

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        # Primer bloque (permite stride si algún día quisieras, aquí usamos 1)
        layers.append(block(self.inplanes, planes, stride=stride))
        self.inplanes = planes * block.expansion
        # Resto de bloques
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        assert x.dim() == 3, f"Esperaba (B,C,T), obtuve {tuple(x.shape)}"
        B, _, T = x.shape

        # Stem
        x = self.relu(self.bn_in(self.conv_in(x)))
        # Body
        x = self.layers(x)
        # Head a C_out manteniendo T
        x = self.head(x)  # (B, C_out, T_actual)

        # Seguridad: igualar exactamente a time_steps si hubiese pequeñas variaciones
        if T != self.time_steps:
            if x.size(-1) > self.time_steps:
                x = x[..., :self.time_steps]
            else:
                pad = self.time_steps - x.size(-1)
                x = nn.functional.pad(x, (0, pad))

        return x  # (B, output_channels, time_steps)
