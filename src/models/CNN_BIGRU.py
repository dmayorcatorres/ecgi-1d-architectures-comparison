import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.ConvSEBlock import ConvSEBlock

class CNN_BIGRU(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 stage1_out: int = 128,
                 stage2_out: int = 128,
                 stage3_out: int = 128,
                 stage4_out: int = 128,
                 gru_hidden: int = 16,
                 dropout_rate: float = 0.8,
                 se_ratio: int = 16,
                 use_se: bool = True):
        super().__init__()
        # ——— Construcción de los bloques CNN usando ConvSEBlock ———
        self.block1 = ConvSEBlock(
            in_channels=input_channels,
            out_channels=stage1_out,
            kernel_size=7,
            se_ratio=se_ratio,
            use_se=use_se
        )
        self.block2 = ConvSEBlock(
            in_channels=stage1_out,
            out_channels=stage2_out,
            kernel_size=5,
            se_ratio=se_ratio,
            use_se=use_se
        )
        self.block3 = ConvSEBlock(
            in_channels=stage2_out,
            out_channels=stage3_out,
            kernel_size=3,
            se_ratio=se_ratio,
            use_se=use_se
        )
        # Bloque 4 también puede usar SE de forma opcional
        self.block4 = ConvSEBlock(
            in_channels=stage3_out,
            out_channels=stage4_out,
            kernel_size=5,
            se_ratio=se_ratio,
            use_se=use_se
        )

        # ——— Rama GRU ———
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True  # Bidireccional
        )
        self.dropout = nn.Dropout(dropout_rate)

        # ——— Capa final 1×1 ———
        self.fc = nn.Conv1d(
            in_channels=stage4_out + 2*gru_hidden,
            out_channels=output_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_channels, T)
        # --- Rama CNN ---
        cnn = self.block1(x)    # (B, stage1_out, T)
        cnn = self.block2(cnn)  # (B, stage2_out, T)
        cnn = self.block3(cnn)  # (B, stage3_out, T)
        cnn = self.block4(cnn)  # (B, stage4_out, T)

        # --- Rama GRU ---
        # Permutar a (B, T, C)
        gru_in = x.permute(0, 2, 1)
        gru_out, _ = self.gru(gru_in)           # (B, T, gru_hidden)
        gru_out = self.dropout(gru_out)
        gru_out = gru_out.permute(0, 2, 1)      # (B, gru_hidden, T)

        # --- Concatenar ambas ramas y salida final ---
        combined = torch.cat([cnn, gru_out], dim=1)  # (B, stage4_out + gru_hidden, T)
        out = self.fc(combined)                     # (B, output_channels, T)
        return out