# src/models/cnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

#############################################
# Modelos 1D para TSC/TSER con hiperparámetros ajustables
#############################################

class SimpleCNNModel(nn.Module):
    """Modelo básico similar a MC-CNN que aplica convoluciones sobre todos los canales.
    
    Parámetros ajustables:
      - conv1_out: número de filtros en la primera capa de convolución.
      - kernel_size_conv1: tamaño del kernel en la primera capa.
      - conv2_out: número de filtros en la segunda capa.
      - kernel_size_conv2: tamaño del kernel en la segunda capa.
      - pool_kernel: tamaño del pooling.
      - upsample_scale: factor de escalado en el upsampling.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 conv1_out: int = 128, kernel_size_conv1: int = 5,
                 conv2_out: int = 64, kernel_size_conv2: int = 3,
                 pool_kernel: int = 2, upsample_scale: int = 2,
                 out_conv_kernel: int = 1) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=conv1_out,
                               kernel_size=kernel_size_conv1,
                               padding=kernel_size_conv1 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        self.conv2 = nn.Conv1d(in_channels=conv1_out,
                               out_channels=conv2_out,
                               kernel_size=kernel_size_conv2,
                               padding=kernel_size_conv2 // 2)
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='nearest')
        self.out_conv = nn.Conv1d(in_channels=conv2_out,
                                  out_channels=output_channels,
                                  kernel_size=out_conv_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class FCNModel(nn.Module):
    """Modelo FCN adaptado para series temporales.
    
    Parámetros ajustables:
      - conv1_out, conv2_out, conv3_out: número de filtros en cada capa.
      - kernel_size_conv1, kernel_size_conv2, kernel_size_conv3: tamaños de kernel.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 conv1_out: int = 128, kernel_size_conv1: int = 7,
                 conv2_out: int = 256, kernel_size_conv2: int = 5,
                 conv3_out: int = 128, kernel_size_conv3: int = 3) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv1d(input_channels, conv1_out,
                               kernel_size=kernel_size_conv1,
                               padding=kernel_size_conv1 // 2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out,
                               kernel_size=kernel_size_conv2,
                               padding=kernel_size_conv2 // 2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.conv3 = nn.Conv1d(conv2_out, conv3_out,
                               kernel_size=kernel_size_conv3,
                               padding=kernel_size_conv3 // 2)
        self.bn3 = nn.BatchNorm1d(conv3_out)
        self.out_conv = nn.Conv1d(conv3_out, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class ResidualBlock(nn.Module):
    """Bloque residual para modelos ResNet.
    
    Parámetros ajustables:
      - kernel_size1, kernel_size2: tamaños de kernel en las dos convoluciones.
    """
    def __init__(self, channels: int, kernel_size1: int = 5, kernel_size2: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return self.relu(x)


class ResNetCNNModel(nn.Module):
    """Modelo ResNet para TSC/TSER.
    
    Parámetros ajustables:
      - num_blocks: número de bloques residuales.
      - conv_in_out: número de filtros en la capa de entrada.
      - kernel_size_in: tamaño del kernel de la capa de entrada.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 num_blocks: int = 3, conv_in_out: int = 64, kernel_size_in: int = 7) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv_in = nn.Conv1d(input_channels, conv_in_out,
                                 kernel_size=kernel_size_in,
                                 padding=kernel_size_in // 2)
        self.bn_in = nn.BatchNorm1d(conv_in_out)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*(ResidualBlock(conv_in_out) for _ in range(num_blocks)))
        self.out_conv = nn.Conv1d(conv_in_out, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class DilatedCNNModel(nn.Module):
    """Modelo con convoluciones dilatadas.
    
    Parámetros ajustables:
      - num_filters: número de filtros en cada capa (por defecto 64).
      - dilations: lista de tasas de dilatación para cada capa.
      - kernel_size: tamaño del kernel (se usa el mismo en todas las capas).
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 num_filters: int = 64, kernel_size: int = 3, dilations: List[int] = [1, 2, 4]) -> None:
        super().__init__()
        self.time_steps = time_steps
        padding = lambda d: (kernel_size - 1) * d // 2
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels if i == 0 else num_filters,
                      num_filters,
                      kernel_size=kernel_size,
                      dilation=d,
                      padding=padding(d))
            for i, d in enumerate(dilations)
        ])
        self.out_conv = nn.Conv1d(num_filters, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class DisjointCNNModel(nn.Module):
    """Modelo Disjoint-CNN que separa procesamiento temporal y espacial.
    
    Parámetros ajustables:
      - temporal_multiplier: factor multiplicador de canales en la convolución temporal.
      - kernel_temporal: tamaño del kernel para la parte temporal.
      - kernel_spatial: tamaño del kernel para la parte espacial.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 temporal_multiplier: int = 2, kernel_temporal: int = 3, kernel_spatial: int = 1) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv_temporal = nn.Conv1d(input_channels, input_channels * temporal_multiplier,
                                       kernel_size=kernel_temporal,
                                       padding=kernel_temporal // 2)
        self.conv_spatial = nn.Conv1d(input_channels * temporal_multiplier, input_channels,
                                       kernel_size=kernel_spatial,
                                       padding=kernel_spatial // 2)
        self.out_conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv_temporal(x))
        x = self.relu(self.conv_spatial(x))
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class MultiScaleCNNModel(nn.Module):
    """Modelo MultiScale-CNN que aplica varias ramas con distintos tamaños de kernel.
    
    Parámetros ajustables:
      - kernel_sizes: lista de tamaños de kernel para cada rama (por defecto [3, 5, 7]).
      - num_filters: número de filtros en cada rama.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 kernel_sizes: List[int] = [3, 5, 7], num_filters: int = 64) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.branches = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(num_filters * len(kernel_sizes))
        self.conv_proj = nn.Conv1d(num_filters * len(kernel_sizes), output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        branch_outputs = [F.relu(conv(x)) for conv in self.branches]
        x_cat = torch.cat(branch_outputs, dim=1)
        x_cat = self.bn(x_cat)
        x_out = self.conv_proj(x_cat)
        return x_out.permute(0, 2, 1)


class MC_DCNN_Model(nn.Module):
    """MC-DCNN: Procesa cada canal de forma independiente con un submodelo.
    
    Parámetros ajustables:
      - sub_channels: número de filtros en el submodelo para cada canal.
      - kernel_size_sub: tamaño del kernel en las capas del submodelo.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 sub_channels: int = 16, kernel_size_sub: int = 3) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.sub_channels = sub_channels
        self.submodel = nn.Sequential(
            nn.Conv1d(1, sub_channels, kernel_size=kernel_size_sub, padding=kernel_size_sub // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(sub_channels, sub_channels, kernel_size=kernel_size_sub, padding=kernel_size_sub // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.proj = nn.Conv1d(sub_channels * input_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        outputs = []
        for c in range(C):
            x_c = x[:, :, c].unsqueeze(1)
            out_c = self.submodel(x_c)
            outputs.append(out_c)
        x_cat = torch.cat(outputs, dim=1)
        x_out = self.proj(x_cat)
        return x_out.permute(0, 2, 1)


class MCCNNModel(nn.Module):
    """MC-CNN / Zhao et al.: Modelo 2-stage.
    
    Parámetros ajustables:
      - stage1_out: número de filtros en la primera etapa.
      - stage2_out: número de filtros en la segunda etapa.
      - kernel_size_stage1 y kernel_size_stage2: tamaños de kernel en cada etapa.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 128, kernel_size_stage1: int = 5,
                 stage2_out: int = 64, kernel_size_stage2: int = 3) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.stage1_conv = nn.Conv1d(input_channels, stage1_out,
                                     kernel_size=kernel_size_stage1,
                                     padding=kernel_size_stage1 // 2)
        self.stage1_relu = nn.ReLU(inplace=True)
        self.stage1_pool = nn.MaxPool1d(kernel_size=2)
        self.stage2_conv = nn.Conv1d(stage1_out, stage2_out,
                                     kernel_size=kernel_size_stage2,
                                     padding=kernel_size_stage2 // 2)
        self.stage2_relu = nn.ReLU(inplace=True)
        self.stage2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_conv = nn.Conv1d(stage2_out, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.stage1_relu(self.stage1_conv(x))
        x = self.stage1_pool(x)
        x = self.stage2_relu(self.stage2_conv(x))
        x = self.stage2_upsample(x)
        x = self.out_conv(x)
        return x.permute(0, 2, 1)


class ResCNNIntegrationModel(nn.Module):
    """Modelo Res-CNN (Zou et al. 2019).
    
    Parámetros ajustables:
      - num_blocks: número de bloques residuales.
      - conv_in_out: número de filtros en la capa de entrada.
      - kernel_size_in: tamaño del kernel en la capa de entrada.
    """
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 num_blocks: int = 3, conv_in_out: int = 64, kernel_size_in: int = 7) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.conv_in = nn.Conv1d(input_channels, conv_in_out,
                                 kernel_size=kernel_size_in,
                                 padding=kernel_size_in // 2)
        self.bn_in = nn.BatchNorm1d(conv_in_out)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*(ResidualBlock(conv_in_out) for _ in range(num_blocks)))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_in_out, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        gap = self.gap(x).squeeze(-1)
        out = self.fc(gap)
        return out.unsqueeze(1).repeat(1, self.time_steps, 1)


class RawCNNBase(nn.Module):
    """Superclase para modelos CNN 1D.
    
    variant: "simple", "fcn", "resnet", "dilated", "disjoint", "multiscale", "mc-dcnn", "mc-cnn", "res-cnn"
    Los hiperparámetros específicos se pueden pasar a cada modelo individualmente.
    """
    def __init__(self, variant: str, time_steps: int, input_channels: int, output_channels: int,
                 **kwargs) -> None:
        super().__init__()
        self.variant = variant.lower()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        if self.variant == "simple":
            self.model = SimpleCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "fcn":
            self.model = FCNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "resnet":
            self.model = ResNetCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "dilated":
            self.model = DilatedCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "disjoint":
            self.model = DisjointCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "multiscale":
            self.model = MultiScaleCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "mc-dcnn":
            self.model = MC_DCNN_Model(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "mc-cnn":
            self.model = MCCNNModel(time_steps, input_channels, output_channels, **kwargs)
        elif self.variant == "res-cnn":
            self.model = ResCNNIntegrationModel(time_steps, input_channels, output_channels, **kwargs)
        else:
            raise ValueError(f"Variante desconocida: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
