# models/cnnbased/zhao.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

def _get_activation(name: str):
    name = name.lower()
    if name in ("relu", "gelu", "tanh", "sigmoid", "selu", "elu"):
        return getattr(F, name)
    raise ValueError(f"Activación no soportada: {name}")

class _ZhaoBackbone(nn.Module):
    """
    Backbone 1D: [Conv-BN-Act-Pool] × 2..3
    - pool_type: 'avg' (paper) o 'max'
    - pool_kernel: 3 (paper sugiere 3)
    - kernel_size_*: 7 (paper sugiere 7)
    """
    def __init__(
        self,
        input_channels: int,
        stage1_out: int = 128,
        kernel_size_stage1: int = 7,
        stage2_out: int = 64,
        kernel_size_stage2: int = 7,
        stage3_out: Optional[int] = None,
        kernel_size_stage3: int = 7,
        pool_kernel: int = 3,
        pool_type: Literal["avg", "max"] = "avg",
        activation: str = "relu",
    ):
        super().__init__()
        self.in_ch = input_channels
        self.use_stage3 = stage3_out is not None
        self.act = _get_activation(activation)

        Pool = nn.AvgPool1d if pool_type == "avg" else nn.MaxPool1d

        # Stage 1
        self.conv1 = nn.Conv1d(input_channels, stage1_out,
                               kernel_size=kernel_size_stage1,
                               padding=kernel_size_stage1 // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(stage1_out)
        self.pool1 = Pool(kernel_size=pool_kernel)

        # Stage 2
        self.conv2 = nn.Conv1d(stage1_out, stage2_out,
                               kernel_size=kernel_size_stage2,
                               padding=kernel_size_stage2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(stage2_out)
        self.pool2 = Pool(kernel_size=pool_kernel)

        # Stage 3 (opcional)
        if self.use_stage3:
            self.conv3 = nn.Conv1d(stage2_out, stage3_out,
                                   kernel_size=kernel_size_stage3,
                                   padding=kernel_size_stage3 // 2, bias=False)
            self.bn3   = nn.BatchNorm1d(stage3_out)
            self.pool3 = Pool(kernel_size=pool_kernel)
            self.final_c = stage3_out
        else:
            self.final_c = stage2_out

        # Inicialización Kaiming
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.use_stage3:
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")

    def _ensure_bct(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Esperaba tensor 3D (B,C,T)/(B,T,C), obtuve {tuple(x.shape)}")
        if x.shape[1] == self.in_ch:
            return x
        if x.shape[2] == self.in_ch:
            return x.permute(0, 2, 1).contiguous()
        raise ValueError(f"Canales no coinciden con input_channels={self.in_ch}")

    def forward(self, x: torch.Tensor):
        x = self._ensure_bct(x)                      # (B, C_in, T)
        x = self.pool1(self.act(self.bn1(self.conv1(x))))
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        if self.use_stage3:
            x = self.pool3(self.act(self.bn3(self.conv3(x))))
        return x  # (B, C_final, T_after)

class ZhaoCNN(nn.Module):
    """
    Misma espina dorsal, pero con cabeza TEMPORAL para regresión secuencia-a-secuencia:
      - 1×1 conv a out_ch en la resolución baja
      - Upsample a 'time_steps' (interpolate) → (B, out_ch, time_steps)
    Úsalo si tu etiqueta es densa en el tiempo (regresión).
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        stage1_out: int = 128,
        kernel_size_stage1: int = 7,
        stage2_out: int = 64,
        kernel_size_stage2: int = 7,
        stage3_out: Optional[int] = None,
        kernel_size_stage3: int = 7,
        pool_kernel: int = 3,
        pool_type: Literal["avg", "max"] = "avg",
        activation: str = "relu",
        upsample_mode: Literal["linear","nearest"] = "linear",
    ):
        super().__init__()
        self.time_steps = time_steps
        self.out_ch = output_channels
        self.upsample_mode = upsample_mode

        self.backbone = _ZhaoBackbone(
            input_channels=input_channels,
            stage1_out=stage1_out, kernel_size_stage1=kernel_size_stage1,
            stage2_out=stage2_out, kernel_size_stage2=kernel_size_stage2,
            stage3_out=stage3_out, kernel_size_stage3=kernel_size_stage3,
            pool_kernel=pool_kernel, pool_type=pool_type, activation=activation
        )
        self.head = nn.Conv1d(self.backbone.final_c, output_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)             # (B, C_final, T_after)
        y = self.head(h)                 # (B, out_ch, T_after)
        # Reescalar exactamente a time_steps (no asume divisibilidad)
        y_up = F.interpolate(y, size=self.time_steps, mode=self.upsample_mode, align_corners=False if self.upsample_mode=="linear" else None)
        return y_up
