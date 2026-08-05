# src/models/VSGRUNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VSGRUNet(nn.Module):
    """
    Variable-Sensitive GRU (VS-GRU)
    -------------------------------------------------------------------
    • Implementa la idea de Li & Xu (2019) con la interfaz mínima que
      exige `ModelFactory`: (time_steps, input_channels, output_channels).
    • Cualquier hiperparámetro adicional llega por **kwargs.
    • Produce salidas paso-a-paso:  (B, C_out, T)
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        **kwargs,
    ) -> None:
        super().__init__()

        # ------------------- hiper-parámetros opcionales -----------------
        hidden_size = kwargs.get("hidden_size", 64)
        num_layers  = kwargs.get("num_layers", 1)
        dropout     = kwargs.get("dropout", 0.0)
        # x==0 solo es un marcador válido de "dato ausente" en datasets que
        # lo codifican así explícitamente. Los datasets de esta pipeline
        # (auckland, bordeaux, ckitpvc, utah2002) son biseñales densas y
        # completas: ~3.7% de las muestras de auckland caen exactamente en
        # 0 por cruces por cero legítimos de la señal, no por ausencia de
        # dato. Por eso el missingness va apagado por defecto; se puede
        # reactivar explícitamente para un dataset que sí tenga huecos
        # codificados como 0.
        self.treat_zero_as_missing = kwargs.get("treat_zero_as_missing", False)
        # ----------------------------------------------------------------

        self.time_steps      = time_steps
        self.input_channels  = input_channels
        self.output_channels = output_channels

        # --- parámetros del missing-factor β  ----------------------------
        self.W_beta = nn.Parameter(torch.randn(input_channels))  # (C,)
        self.b_beta = nn.Parameter(torch.zeros(input_channels))

        # --- media global aprendible por canal (fallback de imputación) ---
        self.x_bar = nn.Parameter(torch.zeros(input_channels))   # (C,)

        # --- GRU que recibe 3·C canales (x̂ || m || β) --------------------
        self.gru = nn.GRU(
            input_size=input_channels * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Conv1d(hidden_size, output_channels, kernel_size=1)

    # ---------------------------------------------------------------------
    @staticmethod
    def _forward_fill(
        x: torch.Tensor,       # (B,C,T)
        m: torch.Tensor,       # (B,C,T)  máscara binaria
        x_bar: torch.Tensor,   # (C,)
    ) -> torch.Tensor:
        """
        Imputación *forward-fill*:
        • copia el último valor observado por canal;
        • si nunca se ha observado, usa la media `x_bar`.
        """
        B, C, T = x.shape
        out  = torch.empty_like(x)
        last = x_bar.view(1, C).repeat(B, 1)          # (B,C)

        for t in range(T):
            cur   = x[:, :, t]                        # (B,C)
            mask  = m[:, :, t].bool()                 # (B,C)
            last  = torch.where(mask, cur, last)      # actualiza últimos
            out[:, :, t] = last                       # asigna imputado
        return out

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_in, T)   →   y : (B, C_out, T)
        """
        B, C, T = x.shape
        assert C == self.input_channels, "input_channels mismatch"
        assert T == self.time_steps,     "time_steps mismatch"

        # 1) Máscara de observación
        if self.treat_zero_as_missing:
            m = (x != 0).float()                        # (B,C,T)
        else:
            m = torch.ones_like(x)                      # todo observado

        # 2) Imputación simple forward-fill
        x_hat = self._forward_fill(x, m, self.x_bar)    # (B,C,T)

        # 3) Missing-rate μ y missing-factor β
        mu   = 1.0 - m.mean(dim=2)                      # (B,C)
        beta = torch.exp(-F.relu(self.W_beta * mu + self.b_beta))  # (B,C)
        beta = beta.unsqueeze(-1).expand(-1, -1, T)     # (B,C,T)

        # 4) Entrada a la GRU: concatena en canales
        inp = torch.cat([x_hat, m, beta], dim=1)        # (B,3C,T)
        gru_in = inp.permute(0, 2, 1)                  # (B,T,3C)

        h_seq, _ = self.gru(gru_in)                    # (B,T,H)
        h_seq = h_seq.permute(0, 2, 1)                 # (B,H,T)

        # 5) Proyección 1×1 → salida
        y = self.head(h_seq)                           # (B,C_out,T)
        return y


# --------------------------- prueba rápida ----------------------------------
if __name__ == "__main__":
    B, C_in, T = 4, 10, 50
    net = VSGRUNet(time_steps=T, input_channels=C_in, output_channels=5,
                   hidden_size=32, num_layers=2, dropout=0.2)

    # crea lote con ~50 % de valores ausentes (0)
    x = torch.randn(B, C_in, T)
    x[torch.rand_like(x).bool()] = 0.0

    y = net(x)
    print("Output shape:", y.shape)   # -> (4, 5, 50)
