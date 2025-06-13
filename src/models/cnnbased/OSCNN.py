# ================================================================
#  OSCNN – Omni-Scale CNN for 1-D Time-Series (ICLR 2022)
# ================================================================
# Mantiene firma    OSCNN(time_steps, input_channels, output_channels, …)
# Compatible con cualquier factory que espere esos tres primeros args.
# ================================================================

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Utilidades  (primos + máscaras de kernels inflados) - CORREGIDO
# ------------------------------------------------------------------
def _prime_list_up_to(n: int) -> List[int]:
    if n < 2:
        return [1]
    sieve = [True] * (n + 1)
    sieve[0:2] = [False, False]  # 0 y 1 no son primos
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = [False] * len(range(p * p, n + 1, p))
    primes = [i for i in range(2, n + 1) if sieve[i]]
    return [1] + primes  # ← incluye 1

def _mask_idx(k_now: int, k_max: int) -> Tuple[int, int]:
    r = math.ceil((k_max - 1) / 2) - math.ceil((k_now - 1) / 2)
    l = k_max - k_now - r
    return l, l + k_now

def _make_mask(cin: int, cout: int, k_now: int, k_max: int) -> np.ndarray:
    mask = np.zeros((cout, cin, k_max), np.float32)
    l, r = _mask_idx(k_now, k_max)
    mask[:, :, l:r] = 1.0
    return mask

def _compose_masks(specs: Sequence[Tuple[int, int, int]]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not specs:
        raise ValueError("OSLayer: 'specs' no puede estar vacío")
    
    # CORRECCIÓN CLAVE: Calcular k_max de forma segura
    k_max = max(spec[-1] for spec in specs)
    
    masks, w0, b0 = [], [], []
    for i, (cin, cout, k) in enumerate(specs):
        conv = nn.Conv1d(cin, cout, k)
        l, r = _mask_idx(k, k_max)
        big_w = np.zeros((cout, cin, k_max), np.float32)
        big_w[:, :, l:r] = conv.weight.detach().cpu().numpy()
        masks.append(_make_mask(cin, cout, k, k_max))
        w0.append(big_w)
        b0.append(conv.bias.detach().cpu().numpy())
    return np.concatenate(masks), np.concatenate(w0), np.concatenate(b0)

# ------------------------------------------------------------------
# 2. Capa Omni-Scale - SIN CAMBIOS
# ------------------------------------------------------------------
class OSLayer(nn.Module):
    def __init__(self, layer_spec: Sequence[Tuple[int, int, int]]):
        super().__init__()
        mask, w0, b0 = _compose_masks(layer_spec)
        C_out, C_in, k_max = mask.shape
        self.register_buffer("mask", torch.from_numpy(mask))
        self.pad = nn.ConstantPad1d(((k_max - 1) // 2, k_max // 2), 0)
        self.conv = nn.Conv1d(C_in, C_out, k_max)
        self.conv.weight = nn.Parameter(torch.from_numpy(w0))
        self.conv.bias = nn.Parameter(torch.from_numpy(b0))
        self.bn = nn.BatchNorm1d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv.weight.data.mul_(self.mask)  # mantiene huecos a cero
        return F.relu(self.bn(self.conv(self.pad(x))))

# ------------------------------------------------------------------
# 3. Bloque Omni-Scale (3 capas) - CORREGIDO
# ------------------------------------------------------------------
class OSBlock(nn.Module):
    def __init__(self, cin: int, base: int, max_rf: int,
                 *, max_primes: Optional[int] = None,
                 bottleneck: Optional[int] = None,
                 dropout: Optional[float] = None):
        super().__init__()

        # Generar y validar lista de primos
        primes = _prime_list_up_to(max_rf)
        total_primes = len(primes)
        
        if max_primes is not None:
            if max_primes < 1:
                raise ValueError(f"max_primes debe ser >=1, recibido {max_primes}")
            # Asegurar que no exceda el número real de primos
            max_primes = min(max_primes, total_primes - 1) if total_primes > 0 else 0
            primes = primes[:max_primes + 1]  # 1 + primeros K primos
        
        if not primes:
            # Fallback seguro para casos extremos
            primes = [1, 2, 3]
            print(f"Advertencia: Usando primos por defecto [1,2,3] para (max_rf={max_rf}, max_primes={max_primes})")

        # Capa 1: múltiples kernels basados en primos
        spec1 = [(cin, base, k) for k in primes]
        mid_c = base * len(primes)

        # Capa de cuello de botella (opcional)
        self.bottle = None
        if bottleneck and bottleneck > 1:
            self.bottle = nn.Conv1d(mid_c, mid_c // bottleneck, 1)
            mid_c //= bottleneck

        # Capa 2 y 3
        spec2 = [(mid_c, base, k) for k in primes]
        spec3 = [(mid_c, base, k) for k in (1, 2)]

        # CORRECCIÓN: Validar que las especificaciones no estén vacías
        for i, spec in enumerate([spec1, spec2, spec3], 1):
            if not spec:
                raise ValueError(f"OSBlock: spec{i} no puede estar vacío (primes={primes})")

        self.layer1 = OSLayer(spec1)
        self.layer2 = OSLayer(spec2)
        self.layer3 = OSLayer(spec3)

        self.out_channels = base * 2
        self.do = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = self.layer1(x)
        if self.bottle is not None:
            x = F.relu(self.bottle(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return self.do(x)

# ------------------------------------------------------------------
# 4. Red completa - SIN CAMBIOS
# ------------------------------------------------------------------
class OSCNN(nn.Module):
    def __init__(self, time_steps: int, input_channels: int,
                 output_channels: int,
                 base_channels: int = 32,
                 max_rf: Optional[int] = None,
                 max_primes: Optional[int] = None,
                 bottleneck: Optional[int] = None,
                 dropout: float = 0.1,
                 global_pool: str = "avg",
                 return_sequence: bool = False):
        super().__init__()

        max_rf = max_rf or time_steps
        if max_rf < 2:
            raise ValueError(f"max_rf debe ser >=2, recibido {max_rf}")
            
        self.return_seq = return_sequence

        self.block = OSBlock(
            input_channels, base_channels, max_rf,
            max_primes=max_primes,
            bottleneck=bottleneck,
            dropout=dropout
        )

        if global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif global_pool == "flatten":
            self.pool = nn.Identity()
        else:
            raise ValueError("global_pool debe ser 'avg', 'max' o 'flatten'")

        feat_dim = (self.block.out_channels if global_pool != "flatten"
                    else self.block.out_channels * time_steps)
        self.fc = nn.Linear(feat_dim, output_channels)
        self.time_steps = time_steps
        self.out_ch = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # (B, T) -> añadir canal
            x = x.unsqueeze(1)
        if x.shape[1] != self.block.layer1.conv.in_channels:
            x = x.permute(0, 2, 1).contiguous()  # (B, C, T)
            
        f = self.block(x)
        if self.return_seq:
            return self.fc(f.transpose(1, 2)).transpose(1, 2)  # (B,out,T)
        g = self.pool(f).squeeze(-1) if f.ndim == 3 else self.pool(f)
        return self.fc(g)  # (B,out)
