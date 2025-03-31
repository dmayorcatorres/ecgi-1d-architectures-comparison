import torch
import torch.nn as nn
import torch.nn.functional as F

class DisjointBlock(nn.Module):
    def __init__(self, in_height, out_filters, temporal_kernel):
        """
        Parámetros:
          - in_height: número de variates (o “altura”) de la entrada (por ejemplo, número de sensores).
          - out_filters: número de filtros de salida (de la parte temporal).
          - temporal_kernel: tamaño del kernel que opera en la dimensión temporal (por ejemplo, 8, 5, o 3).
        """
        super().__init__()
        self.temporal_kernel = temporal_kernel
        # Calcular padding asimétrico para preservar la longitud temporal T.
        if temporal_kernel % 2 == 0:
            self.pad_left = temporal_kernel // 2 - 1
            self.pad_right = temporal_kernel // 2
        else:
            self.pad_left = self.pad_right = temporal_kernel // 2

        # Convolución temporal: opera a lo largo de la dimensión temporal.
        self.temp_conv = nn.Conv2d(in_channels=1,
                                   out_channels=out_filters,
                                   kernel_size=(1, temporal_kernel),
                                   padding=(0, 0))
        self.temp_bn = nn.BatchNorm2d(out_filters)
        self.temp_act = nn.ELU(alpha=1.0)
        
        # Convolución espacial: colapsa la dimensión de variates.
        # Se usa un kernel de tamaño (in_height, 1) para cubrir toda la "altura".
        self.spatial_conv = nn.Conv2d(in_channels=out_filters,
                                      out_channels=out_filters,
                                      kernel_size=(in_height, 1),
                                      padding=0)
        self.spatial_bn = nn.BatchNorm2d(out_filters)
        self.spatial_act = nn.ELU(alpha=1.0)

    def forward(self, x):
        """
        x: Tensor de forma (B, in_height, T)
        Retorna: Tensor de forma (B, out_filters, T)
        """
        # Reestructurar para Conv2d: (B, 1, in_height, T)
        x = x.unsqueeze(1)
        # Aplicar padding asimétrico en la dimensión temporal.
        x = F.pad(x, (self.pad_left, self.pad_right, 0, 0))
        
        # Bloque temporal.
        x = self.temp_conv(x)  # (B, out_filters, in_height, T)
        x = self.temp_bn(x)
        x = self.temp_act(x)
        
        # Bloque espacial: colapsar la dimensión de "in_height".
        x = self.spatial_conv(x)  # (B, out_filters, 1, T)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)
        
        # Eliminar la dimensión "altura" (de tamaño 1).
        x = x.squeeze(2)  # (B, out_filters, T)
        return x

class DisjointLSTM(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 num_filters: int = 32, lstm_hidden: int = 16):
        """
        Arquitectura fusionada que combina:
          - Una rama CNN basada en bloques Disjoint (para extraer patrones locales)
          - Una rama LSTM (para capturar dependencias temporales a largo plazo)
        
        Parámetros:
          - time_steps: longitud temporal T (por ejemplo, 272).
          - input_channels: número de sensores o variates (por ejemplo, 163).
          - output_channels: número de canales de salida (por ejemplo, 502).
          - num_filters: número de filtros que se usan en cada bloque disjoint.
          - lstm_hidden: tamaño del hidden state de la rama LSTM.
        
        La entrada se espera en forma (B, input_channels, T) y la salida en (B, output_channels, T).
        """
        super(DisjointLSTM, self).__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        
        # Rama CNN: se usan 4 bloques disjoint para extraer características locales.
        self.stage1 = DisjointBlock(in_height=input_channels, out_filters=num_filters, temporal_kernel=8)
        self.stage2 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=5)
        self.stage3 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=5)
        self.stage4 = DisjointBlock(in_height=num_filters, out_filters=num_filters, temporal_kernel=3)
        # La salida de la rama CNN tendrá forma (B, num_filters, T)
        
        # Rama LSTM: opera sobre la entrada original transpuesta.
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        # La salida del LSTM será (B, lstm_hidden, T) tras una permutación
        
        # Capa final de fusión: concatenar las dos ramas a lo largo del eje de canales y mapear a output_channels.
        # La concatenación tendrá dimensión: num_filters + lstm_hidden.
        self.final_conv = nn.Conv1d(in_channels=num_filters + lstm_hidden, out_channels=output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor de entrada de forma (B, input_channels, T)
        Retorna: Tensor de salida de forma (B, output_channels, T)
        """
        # Rama CNN: Procesamiento en cascada de bloques disjoint.
        cnn_out = self.stage1(x)
        cnn_out = self.stage2(cnn_out)
        cnn_out = self.stage3(cnn_out)
        cnn_out = self.stage4(cnn_out)  # (B, num_filters, T)
        
        # Rama LSTM:
        # Permutar x para obtener (B, T, input_channels)
        lstm_in = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)  # (B, T, lstm_hidden)
        lstm_out = self.dropout(lstm_out)
        # Permutar la salida de LSTM a (B, lstm_hidden, T)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Fusionar: concatenar las salidas de ambas ramas a lo largo del eje de canales.
        fused = torch.cat([cnn_out, lstm_out], dim=1)  # (B, num_filters + lstm_hidden, T)
        
        # Capa final: mapear la representación fusionada a output_channels.
        out = self.final_conv(fused)  # (B, output_channels, T)
        return out
