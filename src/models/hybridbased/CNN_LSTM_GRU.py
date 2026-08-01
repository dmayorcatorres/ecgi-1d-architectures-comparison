import torch
import torch.nn as nn
from models.base.ConvSEBlock import ConvSEBlock

class CNN_LSTM_GRU(nn.Module):
    """
    Modelo híbrido 1D-CNN + LSTM + GRU para series temporales.
    Combina las salidas de ramas LSTM y GRU con la rama CNN básica.
    Permite activar/desactivar bidireccionalidad por rama.
    """
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 128,
                 stage2_out: int = 128,
                 stage3_out: int = 128,
                 stage4_out: int = 128,
                 lstm_hidden: int = 16,
                 gru_hidden: int = 16,
                 lstm_bidirectional: bool = False,
                 gru_bidirectional: bool = False,
                 dropout_rate: float = 0.2,
                 se_ratio: int = 16,
                 use_se: bool = True):
        super().__init__()
        # ——— Bloques CNN (Conv-BN-ReLU ± SE) ———
        self.blocks = nn.ModuleList([
            ConvSEBlock(input_channels, stage1_out, 7,  se_ratio, use_se),
            ConvSEBlock(stage1_out,    stage2_out, 5,  se_ratio, use_se),
            ConvSEBlock(stage2_out,    stage3_out, 3,  se_ratio, use_se),
            ConvSEBlock(stage3_out,    stage4_out, 5,  se_ratio, use_se),
        ])

        # ——— Rama LSTM ———
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=lstm_bidirectional
        )
        # ——— Rama GRU ———
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=gru_bidirectional
        )

        self.dropout = nn.Dropout(dropout_rate)

        # calcular canales de salida de RNN
        lstm_ch = lstm_hidden * (2 if lstm_bidirectional else 1)
        gru_ch  = gru_hidden  * (2 if gru_bidirectional  else 1)

        # ——— Capa final 1×1 ———
        self.fc = nn.Conv1d(
            in_channels=stage4_out + lstm_ch + gru_ch,
            out_channels=output_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        cnn = x
        for blk in self.blocks:
            cnn = blk(cnn)  # (B, stage*_out, T)

        # — Preparar entrada para RNNs —
        rnn_in = x.permute(0, 2, 1)  # (B, T, C)

        # — LSTM branch —
        lstm_out, _ = self.lstm(rnn_in)    # (B, T, lstm_ch)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, lstm_ch, T)

        # — GRU branch —
        gru_out, _ = self.gru(rnn_in)      # (B, T, gru_ch)
        gru_out = self.dropout(gru_out)
        gru_out = gru_out.permute(0, 2, 1)   # (B, gru_ch, T)

        # — Concatenar y salida —
        combined = torch.cat([cnn, lstm_out, gru_out], dim=1)
        return self.fc(combined)
