# models/mcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

class MultiScaleCNN(nn.Module):
    """
    Multi‑Scale CNN para series temporales (B × C_in × T).
    """

    def __init__(self, time_steps   : int,
        input_channels : int,
        output_channels: int,
        branch_filters : Tuple[int, int, int] | None = None,
        stage1_out: int | None = None,
        stage2_out: int | None = None,
        stage3_out: int | None = None,
        kernel_size : Sequence[int] = (3, 5, 7),
        activation   : str | None    = "relu",
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # 1) Resolver cuántos filtros llevará cada rama
        # ---------------------------------------------------------
        if branch_filters is None:
            # → vienen por separado
            if None in (stage1_out, stage2_out, stage3_out):
                raise ValueError(
                    "Pasa branch_filters=(f1,f2,f3) "
                    "o bien las claves stage{1,2,3}_out"
                )
            branch_filters = (stage1_out, stage2_out, stage3_out)

        if len(branch_filters) != len(kernel_size):
            raise ValueError("branch_filters y kernel_size deben tener la misma longitud")

        self.time_steps = time_steps
        self.act_fn = getattr(F, activation, F.relu)

        # ---------------------------------------------------------
        # 2) Ramas convolucionales
        # ---------------------------------------------------------
        self.branches = nn.ModuleList(
            nn.Conv1d(
                in_channels  = input_channels,
                out_channels = n_filters,
                kernel_size  = k,
                padding      = k // 2
            )
            for k, n_filters in zip(kernel_size, branch_filters)
        )

        self.total_filters = sum(branch_filters)
        self.bn  = nn.BatchNorm1d(self.total_filters)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(self.total_filters, output_channels)

    # -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T)
        if x.shape[1] != self.branches[0].in_channels:
            x = x.permute(0, 2, 1).contiguous()

        feats = torch.cat([self.act_fn(conv(x)) for conv in self.branches], dim=1)
        feats = self.bn(feats)

        z = self.gap(feats).flatten(1)      # (B, C_tot)
        z = self.fc(z)                      # (B, out_ch)

        return z.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, out_ch, T)
