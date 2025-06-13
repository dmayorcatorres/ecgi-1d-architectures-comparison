import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#   InceptionParallel
##############################################################################

class InceptionParallel(nn.Module):
    """
    Bloque Inception que aplica varias convoluciones en paralelo (multiplexing).
    - Adicionalmente, puede incluir "custom filters" (filtros fijos no entrenables).
    - Usa 'padding=same' (solo PyTorch >= 2.0 con stride=1).
    """

    def __init__(self,
                 input_channels: int,
                 stage1_out: int = 32,
                 base_kernel: int = 20,
                 use_custom_filters: bool = False,
                 dilation_rate: int = 1):
        """
        Args:
            input_channels (int): número de canales de entrada.
            stage1_out (int): número de filtros por rama.
            base_kernel (int): tamaño base (máximo) del kernel. Se generarán 3 kernels decrecientes.
            use_custom_filters (bool): si `True`, añade ramas con filtros fijos (no entrenables).
            dilation_rate (int): dilatación a usar en las convoluciones estándar.
        """
        super(InceptionParallel, self).__init__()

        # Ejemplo: 3 tamaños de kernel calculados
        self.kernel_size_list = [base_kernel // (2**i) for i in range(3)]

        # --- Ramas estándar ---
        self.standard_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=stage1_out,
                kernel_size=k,
                padding='same',      # PyTorch >= 2.0
                stride=1,
                dilation=dilation_rate,
                bias=True
            )
            for k in self.kernel_size_list
        ])

        # --- Ramas "custom" (opcional) ---
        self.use_custom_filters = use_custom_filters
        if use_custom_filters:
            self.custom_convs = nn.ModuleList()

            # 3 conv con los mismos kernel_size
            for k in self.kernel_size_list:
                conv = nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=stage1_out,
                    kernel_size=k,
                    padding='same',
                    stride=1,
                    bias=False
                )
                # "Fijar" pesos => no se entrenan
                conv.weight.requires_grad = False
                self.custom_convs.append(conv)

            # 2 conv extras => (k + k//2) en kernel
            for k in self.kernel_size_list[1:]:
                k2 = k + (k // 2)
                conv = nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=stage1_out,
                    kernel_size=k2,
                    padding='same',
                    stride=1,
                    bias=False
                )
                conv.weight.requires_grad = False
                self.custom_convs.append(conv)
        else:
            self.custom_convs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_channels, time)
        Retorna: (batch_size, #ramas*stage1_out, time)
        """
        out_branches = []

        # Ramas estándar
        for conv in self.standard_convs:
            out_branches.append(conv(x))

        # Ramas custom, si existen
        if self.use_custom_filters and self.custom_convs is not None:
            for conv in self.custom_convs:
                out_branches.append(conv(x))

        # Concatenar salidas por eje de canales
        out = torch.cat(out_branches, dim=1)
        return out


##############################################################################
#   LITE (Light Inception with boosting tEchniques)
##############################################################################

class LITE(nn.Module):
    """
    Arquitectura LITE:
      - Primera capa: InceptionParallel (multiplexing + custom filters)
      - 2 capas de Depthwise Separable Convolution (DWSC) con BatchNorm
      - Convolución final => (B, output_channels, T)
    """

    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 32,
                 base_kernel_size: int = 41,
                 use_custom_filters: bool = True):
        """
        Args:
            input_channels (int): # de canales de entrada (por ej. univariante => 1).
            time_steps (int): longitud temporal (para referencia).
            output_channels (int): # de canales de salida (clases, o embeddings).
            stage1_out (int): # de filtros tras la capa de inception.
            base_kernel_size (int): tamaño base (k) para la 1era capa.
            use_custom_filters (bool): si True, añade filtros fijos no entrenables.
        """
        super(LITE, self).__init__()

        self.input_channels = input_channels
        self.time_steps = time_steps
        self.output_channels = output_channels
        self.stage1_out = stage1_out
        # Ajuste: base_kernel = base_kernel_size - 1 (como en la versión original)
        self.base_kernel = base_kernel_size - 1

        # 1) Inception parallel
        self.inception = InceptionParallel(
            input_channels=input_channels,
            stage1_out=stage1_out,
            base_kernel=self.base_kernel,
            use_custom_filters=use_custom_filters,
            dilation_rate=1
        )

        # Calcular # de canales tras la concatenación
        if use_custom_filters:
            # 3 ramas "standard" + 5 ramas "custom" = 8 * stage1_out
            inception_out = stage1_out * (3 + 5)
        else:
            # solo 3 ramas => 3 * stage1_out
            inception_out = stage1_out * 3

        # 2) Primera capa Depthwise Separable
        k1 = max(3, self.base_kernel // 2)  # ejemplo
        self.dwsc1 = nn.Conv1d(
            in_channels=inception_out,
            out_channels=inception_out,  # depthwise => #out=#in
            kernel_size=k1,
            padding='same',   # PyTorch >=2.0
            stride=1,
            groups=inception_out,  # depthwise
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(inception_out)
        # Pointwise => reduce a stage1_out
        self.pw1 = nn.Conv1d(in_channels=inception_out,
                             out_channels=stage1_out,
                             kernel_size=1)

        # 3) Segunda capa Depthwise Separable (con dilation=4)
        k2 = max(3, self.base_kernel // 4)
        self.dwsc2 = nn.Conv1d(
            in_channels=stage1_out,
            out_channels=stage1_out,
            kernel_size=k2,
            padding='same',
            stride=1,
            groups=stage1_out,   # depthwise
            dilation=4,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(stage1_out)
        self.pw2 = nn.Conv1d(in_channels=stage1_out,
                             out_channels=stage1_out,
                             kernel_size=1)

        # 4) Convolución final => map stage1_out -> output_channels
        #    conservando dimension T
        self.final_conv = nn.Conv1d(
            in_channels=stage1_out,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (batch_size, input_channels, time_steps)
        Salida => (batch_size, output_channels, time_steps)
        """
        # (A) Inception
        x = self.inception(x)
        x = F.relu(x)

        # (B) Depthwise #1
        x = self.dwsc1(x)      # depthwise
        x = self.bn1(x)
        x = self.pw1(x)        # pointwise
        x = F.relu(x)

        # (C) Depthwise #2
        x = self.dwsc2(x)
        x = self.bn2(x)
        x = self.pw2(x)
        x = F.relu(x)

        # (D) Convolución final
        x = self.final_conv(x)  # => (B, output_channels, T)
        return x


""" ##############################################################################
#   EJEMPLO DE USO
##############################################################################
if __name__ == '__main__':
    # Datos de prueba
    batch_size = 8
    input_channels = 163   # por ejemplo ECG con 163 leads
    time_steps = 272       # # de muestras temporales
    output_channels = 502  # e.g. # de clases, o canales en la salida

    # Forma => (batch_size, input_channels, time_steps)
    x = torch.randn(batch_size, input_channels, time_steps)

    # Crear modelo LITE
    model = LITE(
        input_channels=input_channels,
        time_steps=time_steps,
        output_channels=output_channels,
        stage1_out=32,
        base_kernel_size=41,
        use_custom_filters=False
    )

    # Inferencia
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")  # esperado: [8, 502,
 """