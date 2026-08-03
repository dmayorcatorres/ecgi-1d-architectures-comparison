# models/cnnbased/MultiScaleCNN.py
import torch
import torch.nn as nn
from typing import Sequence, Tuple

def _make_act(name: str | None) -> nn.Module:
    name = (name or "relu").lower()
    if name == "gelu": return nn.GELU()
    if name in ("silu", "swish"): return nn.SiLU()
    if name in ("lrelu", "leaky_relu"): return nn.LeakyReLU(0.01, inplace=True)
    if name == "elu": return nn.ELU(inplace=True)
    return nn.ReLU(inplace=True)

class _Branch1D(nn.Module):
    """Una rama: Conv(k) → BN → Act. Mantiene T con padding=k//2."""
    def __init__(self, in_ch: int, out_ch: int, k: int, act: nn.Module):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = act
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MultiScaleCNN(nn.Module):
    """
    Multi-Scale 1D CNN (entrada y salida con misma resolución temporal):
      ramas en paralelo con kernels distintos → concat (canales) →
      fusión 1×1 → cabeza 1×1 a output_channels.

    Firma compatible con tu HPO:
      - time_steps, input_channels, output_channels
      - stage1_out, stage2_out, stage3_out (o branch_filters)
      - kernel_size: p.ej. (3,5,7)
      - activation: string (relu/gelu/silu/elu/lrelu)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        branch_filters: Tuple[int, int, int] | None = None,
        stage1_out: int | None = None,
        stage2_out: int | None = None,
        stage3_out: int | None = None,
        kernel_size: Sequence[int] = (3, 5, 7),
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        # 1) Resolver nº de filtros por rama
        if branch_filters is None:
            if None in (stage1_out, stage2_out, stage3_out):
                raise ValueError("Pasa branch_filters=(f1,f2,f3) o stage{1,2,3}_out")
            branch_filters = (stage1_out, stage2_out, stage3_out)
        if len(branch_filters) != len(kernel_size):
            raise ValueError("branch_filters y kernel_size deben tener la misma longitud")

        self.time_steps = time_steps
        self.in_ch  = input_channels
        self.out_ch = output_channels
        act = _make_act(activation)

        # 2) Ramas paralelas (cada una Conv→BN→Act, preserva T)
        self.branches = nn.ModuleList([
            _Branch1D(input_channels, nf, k, act) for k, nf in zip(kernel_size, branch_filters)
        ])
        self.total_filters = sum(branch_filters)

        # 3) Fusión 1×1 (mezcla de ramas) + 4) Cabeza 1×1 a C_out
        self.fuse = nn.Sequential(
            nn.Conv1d(self.total_filters, self.total_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.total_filters),
            act,
        )
        self.head = nn.Conv1d(self.total_filters, output_channels, kernel_size=1, bias=True)

        nn.init.kaiming_normal_(self.fuse[0].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")

    def _ensure_bct(self, x: torch.Tensor) -> torch.Tensor:
        # Acepta (B,C,T) o (B,T,C); normaliza a (B,C,T)
        if x.dim() != 3:
            raise ValueError(f"Esperaba tensor 3D (B,C,T)/(B,T,C), obtuve {tuple(x.shape)}")
        if x.shape[1] == self.in_ch:
            return x
        if x.shape[2] == self.in_ch:
            return x.permute(0, 2, 1).contiguous()
        raise ValueError(f"Canales no coinciden con input_channels={self.in_ch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_bct(x)                              # (B, C_in, T)
        feats = torch.cat([b(x) for b in self.branches], 1)  # (B, ΣF, T)
        feats = self.fuse(feats)                             # (B, ΣF, T)
        out   = self.head(feats)                             # (B, C_out, T)

        # Seguridad: igualar exactamente a time_steps si hiciera falta
        if out.size(-1) != self.time_steps:
            T = out.size(-1)
            if T > self.time_steps:
                out = out[..., :self.time_steps]
            else:
                out = nn.functional.pad(out, (0, self.time_steps - T))
        return out  # (B, output_channels, time_steps)
