import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from src.models.base.DisjointBase import DisjointBase

########################################
# Híbrido: DisjointLSTM
########################################
class DisjointLSTM(nn.Module):
    def __init__(self,
                 time_steps: int,
                 input_channels: int,
                 output_channels: int,
                 num_filters: int = 32,
                 lstm_hidden: int = 16,
                 lstm_layers: int = 1,
                 dropout_prob: float = 0.8,
                 kernels: List[int] = [8, 5, 5, 3]):
        """
        Modelo híbrido que fusiona:
          - Una rama CNN basada en bloques DisjointBase.
          - Una rama LSTM que opera sobre la entrada original.
        
        Parámetros:
          - time_steps: longitud temporal T.
          - input_channels: número de variates (sensores) de la entrada.
          - output_channels: número de canales de salida.
          - num_filters: número de filtros en cada bloque CNN.
          - lstm_hidden: tamaño del estado oculto del LSTM.
          - lstm_layers: cantidad de capas del LSTM.
          - dropout_prob: probabilidad de dropout tras el LSTM.
          - kernels: lista de tamaños de kernel para cada etapa del bloque disjoint.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Rama CNN: secuencia de bloques disjoint.
        self.cnn_layers = nn.ModuleList()
        # Primer bloque: opera sobre la entrada original.
        self.cnn_layers.append(DisjointBase(in_height=input_channels, out_filters=num_filters, temporal_kernel=kernels[0]))
        # Bloques siguientes: la "altura" pasa a ser num_filters.
        for kernel in kernels[1:]:
            self.cnn_layers.append(DisjointBase(in_height=num_filters, out_filters=num_filters, temporal_kernel=kernel))
            
        # Rama LSTM: se procesa la entrada original.
        # Se permuta la entrada a forma (B, T, input_channels)
        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Capa final de fusión: concatenación de la salida de la rama CNN y la LSTM.
        # La salida será de dimensión (num_filters + lstm_hidden) en el eje de canales.
        self.final_conv = nn.Conv1d(in_channels=num_filters + lstm_hidden,
                                    out_channels=output_channels,
                                    kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
          - x: Tensor de entrada con forma (B, input_channels, T)
        Retorna:
          - Tensor de salida con forma (B, output_channels, T)
        """
        # Rama CNN: procesamiento en cascada usando los bloques DisjointBase.
        cnn_out = x
        for layer in self.cnn_layers:
            cnn_out = layer(cnn_out)
            # Cada bloque transforma la entrada a (B, num_filters, T)
        
        # Rama LSTM:
        # Permutar x a (B, T, input_channels)
        lstm_in = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)  # (B, T, lstm_hidden)
        lstm_out = self.dropout(lstm_out)
        # Volver a permutar la salida a (B, lstm_hidden, T)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Fusionar ambas ramas: concatenar a lo largo del eje de canales.
        fused = torch.cat((cnn_out, lstm_out), dim=1)  # (B, num_filters + lstm_hidden, T)
        # Capa final para mapear a output_channels.
        out = self.final_conv(fused)
        return out