import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.ConvSEBlock import ConvSEBlock

class CNN_BILSTM(nn.Module):
    """
    Modelo híbrido 1D-CNN + LSTM para series temporales.

    Parámetros:
        time_steps: longitud temporal de entrada (T).
        input_channels: número de canales de entrada (C).
        output_channels: número de canales de salida.
        stageX_out: filtros de cada bloque CNN.
        lstm_hidden: dimensión de hidden state de LSTM.
        dropout_rate: probabilidad de dropout en la rama LSTM.
        se_ratio: ratio del bloque SE.
        use_se: activar o desactivar bloques SE.
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
                 dropout_rate: float = 0.8,
                 se_ratio: int = 16,
                 use_se: bool = True):
        super().__init__()
        # Bloques CNN con opciones SE
        self.block1 = ConvSEBlock(input_channels, stage1_out, kernel_size=7, se_ratio=se_ratio, use_se=use_se)
        self.block2 = ConvSEBlock(stage1_out,    stage2_out, kernel_size=5, se_ratio=se_ratio, use_se=use_se)
        self.block3 = ConvSEBlock(stage2_out,    stage3_out, kernel_size=3, se_ratio=se_ratio, use_se=use_se)
        self.block4 = ConvSEBlock(stage3_out,    stage4_out, kernel_size=5, se_ratio=se_ratio, use_se=use_se)

        # Rama LSTM
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True  # Bidireccional
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Capa final 1x1 conv
        self.fc = nn.Conv1d(
            in_channels=stage4_out + 2*lstm_hidden,
            out_channels=output_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # CNN branch
        cnn = self.block1(x)    # (B, stage1_out, T)
        cnn = self.block2(cnn)  # (B, stage2_out, T)
        cnn = self.block3(cnn)  # (B, stage3_out, T)
        cnn = self.block4(cnn)  # (B, stage4_out, T)

        # LSTM branch: permutar a (B, T, C)
        lstm_in = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)       # (B, T, lstm_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)   # (B, lstm_hidden, T)

        # Concatenar y salida
        combined = torch.cat([cnn, lstm_out], dim=1)  # (B, stage4_out+lstm_hidden, T)
        out = self.fc(combined)                        # (B, output_channels, T)
        return out
