# src/models/densenetbilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


# ───────────────── 1) bloques auxiliares ───────────────────────────
class _DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth: int, n_layers: int, k_size: int):
        super().__init__()
        self.k = k_size
        mods = []
        for _ in range(n_layers):
            mods.append(nn.BatchNorm1d(in_ch))
            mods.append(nn.ReLU(inplace=True))
            mods.append(nn.Conv1d(in_ch, growth, k_size, bias=False))
            in_ch += growth
        self.block = nn.Sequential(*mods)
        self.out_ch = in_ch

    def _same_pad(self, x):
        l = (self.k - 1) // 2
        r = self.k - 1 - l
        return F.pad(x, (l, r))

    def forward(self, x):
        out = x
        for i in range(0, len(self.block), 3):         # BN‑ReLU‑Conv
            y = self._same_pad(out)
            y = self.block[i : i + 3](y)
            out = torch.cat([out, y], dim=1)
        return out


class _Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: int = 2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.MaxPool1d(pool, pool)
        )

    def forward(self, x):
        return self.seq(x)


# ───────────────── 2) rama única ───────────────────────────────────
class _ECGBranch(nn.Module):
    def __init__(self,
                 in_ch: int,
                 init_filters: int,
                 growth_rates: Sequence[int],
                 n_layers: Sequence[int],
                 k_sizes: Sequence[int],
                 trans_ch: Sequence[int],
                 lstm_hidden: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, init_filters, 32, padding=15, bias=False),
            nn.BatchNorm1d(init_filters),
            nn.ReLU(inplace=True)
        )
        c = init_filters

        self.db1 = _DenseBlock(c, growth_rates[0], n_layers[0], k_sizes[0]); c = self.db1.out_ch
        self.tr1 = _Transition(c, trans_ch[0]);                              c = trans_ch[0]
        self.db2 = _DenseBlock(c, growth_rates[1], n_layers[1], k_sizes[1]); c = self.db2.out_ch
        self.tr2 = _Transition(c, trans_ch[1]);                              c = trans_ch[1]

        self.rnn = nn.LSTM(
            input_size=c,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.tr1(self.db1(x))
        x = self.tr2(self.db2(x))
        x = x.permute(0, 2, 1)                # (B,T,C) para LSTM
        x, _ = self.rnn(x)
        x = self.gap(x.permute(0, 2, 1)).squeeze(-1)
        return self.drop(x)


# ───────────────── 3) modelo completo ──────────────────────────────
class DenseNetBiLSTM(nn.Module):
    """
    Acepta tanto los parámetros ‘compactos’ (listas) como los
    ‘flat’ que genera Optuna: growth_rate_small, num_layers_large, …
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 *,
                 # comunes
                 initial_filters: int = 16,
                 kernel_sizes: Sequence[int] = (32, 32),
                 lstm_hidden: int = 128,
                 # ① formato “compacto” (listas)  ────────────────
                 growth_rates: Sequence[int] | None = None,
                 num_layers:   Sequence[int] | None = None,
                 transition_channels: Sequence[int] | None = None,
                 # ② formato “flat”  ─────────────────────────────
                 growth_rate_small: int | None = None,
                 growth_rate_large: int | None = None,
                 num_layers_small:  int | None = None,
                 num_layers_large:  int | None = None,
                 transition_small:  int | None = None,
                 transition_large:  int | None = None):
        super().__init__()

        # ── Normalizar a listas -------------------------------------------------
        if growth_rates is None:
            growth_rates = [growth_rate_small, growth_rate_large]
        if num_layers is None:
            num_layers = [num_layers_small, num_layers_large]
        if transition_channels is None:
            transition_channels = [transition_small, transition_large]

        # ── dos ramas idénticas -------------------------------------------------
        self.branch_small = _ECGBranch(
            input_channels, initial_filters,
            growth_rates, num_layers, kernel_sizes,
            transition_channels, lstm_hidden
        )
        self.branch_large = _ECGBranch(
            input_channels, initial_filters,
            growth_rates, num_layers, kernel_sizes,
            transition_channels, lstm_hidden
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * lstm_hidden, 64),   # 2 ramas × 2*hidden
            nn.ReLU(inplace=True),
            nn.Linear(64, output_channels)
        )

    def forward(self, x):                    # Ignite pasa (B,C,T)
        fs = self.branch_small(x)
        fl = self.branch_large(x)
        return self.classifier(torch.cat([fs, fl], dim=1))
