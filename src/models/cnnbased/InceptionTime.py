import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_channels=32, use_residual=True):
        """
        Parámetros:
          - in_channels: número de canales de entrada.
          - nb_filters: número de filtros a usar en cada rama.
          - kernel_sizes: lista de longitudes de filtro.
          - bottleneck_channels: número de canales en la capa de bottleneck (conv 1x1).
          - use_residual: si se utiliza la conexión residual.
        """
        super(InceptionModule, self).__init__()
        # Bottleneck: si in_channels > 1, se reduce la dimensionalidad.
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        else:
            bottleneck_channels = in_channels

        self.kernel_sizes = kernel_sizes
        # Calcular el padding "same" para cada tamaño de filtro.
        self.pad_params = {}
        for k in kernel_sizes:
            total_padding = k - 1  # Para stride=1.
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            self.pad_params[k] = (pad_left, pad_right)

        # Ramas convolucionales: se aplicará padding manualmente.
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_branches.append(
                nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=k, padding=0, bias=False)
            )

        # Rama de MaxPooling seguida de conv1d 1x1.
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        )

        # Concatenación: número total de canales = nb_filters * (n_ramas + 1).
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

        # Conexión residual.
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, nb_filters * (len(kernel_sizes) + 1), kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))

    def forward(self, x):
        # x: (B, in_channels, T)
        if self.use_bottleneck:
            x_reduced = self.bottleneck(x)  # (B, bottleneck_channels, T)
        else:
            x_reduced = x

        branch_outputs = []
        for idx, conv in enumerate(self.conv_branches):
            k = self.kernel_sizes[idx]
            pad_left, pad_right = self.pad_params[k]
            padded = F.pad(x_reduced, (pad_left, pad_right))
            branch_outputs.append(conv(padded))  # Cada salida: (B, nb_filters, T)

        branch_outputs.append(self.maxpool_branch(x))  # (B, nb_filters, T)
        out = torch.cat(branch_outputs, dim=1)  # (B, nb_filters*(n_branches), T)
        out = self.bn(out)
        out = self.relu(out)

        if self.use_residual:
            residual = self.residual_conv(x)
            residual = self.residual_bn(residual)
            out = self.relu(out + residual)
        return out

class InceptionTime(nn.Module):
    def __init__(self, time_steps, input_channels, output_channels,
                 stage1_out=32, kernel_sizes=[10, 20, 40], bottleneck_channels=32,
                 use_residual=True, num_inception_modules=6):
        """
        Parámetros:
          - time_steps: longitud temporal (por ejemplo, 100).
          - input_channels: número de canales de entrada (por ejemplo, 163).
          - output_channels: número de canales de salida deseados (por ejemplo, 502).
          - num_inception_modules: cantidad de módulos Inception a apilar.
        """
        super(InceptionTime, self).__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_inception_modules = num_inception_modules

        # Primer módulo: toma la entrada original.
        self.inception_modules = nn.ModuleList()
        self.inception_modules.append(
            InceptionModule(in_channels=input_channels, nb_filters=stage1_out,
                            kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels,
                            use_residual=use_residual)
        )
        # Los módulos posteriores toman la salida del anterior.
        for _ in range(1, num_inception_modules):
            in_ch = stage1_out * (len(kernel_sizes) + 1)
            self.inception_modules.append(
                InceptionModule(in_channels=in_ch, nb_filters=stage1_out,
                                kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels,
                                use_residual=use_residual)
            )
        
        # Capa final para mapear a output_channels.
        final_in_channels = stage1_out * (len(kernel_sizes) + 1)
        self.final_conv = nn.Conv1d(in_channels=final_in_channels, out_channels=output_channels,
                                    kernel_size=1, bias=False)
    
    def forward(self, x):
        # x: (B, input_channels, T)
        out = x
        for module in self.inception_modules:
            out = module(out)
        out = self.final_conv(out)  # (B, output_channels, T)
        return out