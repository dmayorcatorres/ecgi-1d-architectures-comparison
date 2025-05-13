# ================================================================
#  OSCNN – implementación basada en “Omni‑Scale CNNs” (ICLR 2022)
# ================================================================
# Mantiene la signatura  __init__(time_steps, input_channels, output_channels,…)
# para que tu factory lo invoque sin problemas.
#
# Correcciones clave:
#   •  self.os_block.out_channels  → número real de features de salida
#   •  Linear(in_features = os_block.out_channels, …)
# ================================================================

from __future__ import annotations
import math
from typing import List, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------
# 1.  utilidades para crear la máscara de kernels “inflados”
# ------------------------------------------------------------------
def _mask_indices(k_now: int, k_max: int) -> Tuple[int, int]:
    right_pad = math.ceil((k_max - 1) / 2) - math.ceil((k_now - 1) / 2)
    left_pad  = k_max - k_now - right_pad
    return left_pad, left_pad + k_now


def _make_kernel_mask(cin: int, cout: int,
                      k_now: int, k_max: int) -> np.ndarray:
    l, r = _mask_indices(k_now, k_max)
    mask = np.zeros((cout, cin, k_max), dtype=np.float32)
    mask[:, :, l:r] = 1.0
    return mask


def _compose_layer_masks(params: Sequence[Tuple[int, int, int]]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recibe la lista de kernels (Cin, Cout, k) de una misma capa y devuelve:
        • máscara 0/1
        • pesos iniciales “inflados”
        • bias concatenado
    """
    k_max = params[-1][-1]
    masks, w_init, b_init = [], [], []

    for cin, cout, k in params:
        conv = nn.Conv1d(cin, cout, k)                # init He
        l, r = _mask_indices(k, k_max)
        big_w = np.zeros((cout, cin, k_max), np.float32)
        big_w[:, :, l:r] = conv.weight.detach().cpu().numpy()

        masks.append(_make_kernel_mask(cin, cout, k, k_max))
        w_init.append(big_w)
        b_init.append(conv.bias.detach().cpu().numpy())

    return (np.concatenate(masks, axis=0),
            np.concatenate(w_init, axis=0),
            np.concatenate(b_init, axis=0))


# ------------------------------------------------------------------
# 2. Capa Omni‑Scale (kernels primos fusionados en una Conv1d)
# ------------------------------------------------------------------
class OSLayer(nn.Module):
    def __init__(self,
                 layer_spec: Sequence[Tuple[int, int, int]]):
        super().__init__()

        mask, w0, b0 = _compose_layer_masks(layer_spec)
        cout, cin, k_max = mask.shape

        self.register_buffer('weight_mask', torch.from_numpy(mask))

        self.pad  = nn.ConstantPad1d(((k_max - 1) // 2, k_max // 2), 0)
        self.conv = nn.Conv1d(cin, cout, k_max)
        self.conv.weight = nn.Parameter(torch.from_numpy(w0))
        self.conv.bias   = nn.Parameter(torch.from_numpy(b0))

        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enmascaramos en cada forward para mantener huecos a cero
        self.conv.weight.data *= self.weight_mask
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


# ------------------------------------------------------------------
# 3. Bloque Omni‑Scale (3 OSLayer)
# ------------------------------------------------------------------
def _prime_list_up_to(n: int) -> List[int]:
    sieve = [True] * (n + 1)
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:n + 1:p] = [False] * len(range(p * p, n + 1, p))
    return [2] + [i for i in range(3, n + 1, 2) if sieve[i]]


class OSBlock(nn.Module):
    """
    • Capas 1‑2: kernels primos 1..p_k
    • Capa   3: kernels {1, 2}
    """
    def __init__(self,
                 cin: int,
                 base_channels: int,
                 max_rf: int):
        super().__init__()

        primes = _prime_list_up_to(max_rf)

        # especificaciones (Cin, Cout, kernel)
        layer1 = [(cin, base_channels, k) for k in primes]
        layer2 = [(base_channels * len(primes), base_channels, k) for k in primes]
        layer3 = [(base_channels * len(primes), base_channels, k) for k in (1, 2)]

        self.layer1 = OSLayer(layer1)
        self.layer2 = OSLayer(layer2)
        self.layer3 = OSLayer(layer3)

        # ▸ salen exactamente 2 × base_channels (kernels 1 y 2)
        self.out_channels = base_channels * 2

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


# ------------------------------------------------------------------
# 4. Clasificador completo
# ------------------------------------------------------------------
# ───────── en OSCNN.py ──────────────────────────────────────────
class OSBlock(nn.Module):
    def __init__(self,
                 cin: int,
                 base_channels: int,
                 max_rf: int,
                 max_primes: int | None = None,   # <── NEW
                 bottleneck: int | None = None):   # <── NEW
        super().__init__()

        primes_all = _prime_list_up_to(max_rf)
        if max_primes:                # ≤ por memoria
            primes = primes_all[:max_primes]
        else:
            primes = primes_all

        # ─ capa 1
        layer1 = [(cin, base_channels, k) for k in primes]

        # ─ opcional: cuello de botella 1×1
        mid_channels = base_channels * len(primes)
        if bottleneck:                # e.g. bottleneck = 4  ⇒  /4 canales
            self.bottle = nn.Conv1d(mid_channels,
                                    mid_channels // bottleneck, 1)
            mid_channels //= bottleneck
        else:
            self.bottle = None

        # ─ capa 2 y 3
        layer2 = [(mid_channels, base_channels, k) for k in primes]
        layer3 = [(mid_channels, base_channels, k) for k in (1, 2)]

        self.layer1 = OSLayer(layer1)
        self.layer2 = OSLayer(layer2)
        self.layer3 = OSLayer(layer3)

        self.out_channels = base_channels * 2

    def forward(self, x):
        x = self.layer1(x)
        if self.bottle is not None:
            x = F.relu(self.bottle(x))
        x = self.layer2(x)
        return self.layer3(x)