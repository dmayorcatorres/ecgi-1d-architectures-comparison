import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

class DisjointResNet(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int, n_feature_maps: int = 32):
        """
        Modelo DisjointResNetMIMO secuencial para reconstrucción ECGi.
        
        Esta implementación sigue la estructura secuencial con fusión en cada bloque
        como en Classifier_D_ResNet.
        
        Parámetros:
          - time_steps: longitud temporal de la serie.
          - input_channels: número de canales de entrada (sensores ECG en torso).
          - output_channels: número de canales de salida (electrodos epicárdicos).
          - n_feature_maps: número de filtros base en las capas convolucionales.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_feature_maps = n_feature_maps
        
        # BLOQUE 1
        # Convoluciones Temporales - Camino Principal
        self.conv1_x_temporal = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=8,
            padding='same',
            groups=input_channels  # Convolution separable por canal
        )
        self.bn1_x_temporal = nn.BatchNorm1d(input_channels)
        
        # Convoluciones Espaciales - Camino Principal
        self.conv1_x_spatial = nn.Conv1d(
            in_channels=input_channels,
            out_channels=n_feature_maps,
            kernel_size=1  # Convolución puntual
        )
        self.bn1_x_spatial = nn.BatchNorm1d(n_feature_maps)
        
        # Convoluciones Temporales - Segunda Capa
        self.conv1_y_temporal = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=5,
            padding='same',
            groups=n_feature_maps  # Convolution separable por canal
        )
        self.bn1_y_temporal = nn.BatchNorm1d(n_feature_maps)
        
        # Convoluciones Espaciales - Segunda Capa
        self.conv1_y_spatial = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=1  # Convolución puntual
        )
        self.bn1_y_spatial = nn.BatchNorm1d(n_feature_maps)
        
        # Convoluciones Temporales - Tercera Capa
        self.conv1_z_temporal = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=3,
            padding='same',
            groups=n_feature_maps  # Convolution separable por canal
        )
        self.bn1_z_temporal = nn.BatchNorm1d(n_feature_maps)
        
        # Convoluciones Espaciales - Tercera Capa
        self.conv1_z_spatial = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=1  # Convolución puntual
        )
        self.bn1_z_spatial = nn.BatchNorm1d(n_feature_maps)
        
        # Camino de Atajo para Bloque 1
        self.shortcut1_temporal = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            padding='same'
        )
        self.bn_shortcut1_temporal = nn.BatchNorm1d(input_channels)
        
        self.shortcut1_spatial = nn.Conv1d(
            in_channels=input_channels,
            out_channels=n_feature_maps,
            kernel_size=1
        )
        self.bn_shortcut1_spatial = nn.BatchNorm1d(n_feature_maps)
        
        # BLOQUE 2
        # Convoluciones Temporales - Camino Principal
        self.conv2_x_temporal = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=8,
            padding='same',
            groups=n_feature_maps  # Convolution separable por canal
        )
        self.bn2_x_temporal = nn.BatchNorm1d(n_feature_maps)
        
        # Convoluciones Espaciales - Camino Principal
        self.conv2_x_spatial = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn2_x_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Temporales - Segunda Capa
        self.conv2_y_temporal = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=5,
            padding='same',
            groups=n_feature_maps * 2  # Convolution separable por canal
        )
        self.bn2_y_temporal = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Espaciales - Segunda Capa
        self.conv2_y_spatial = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn2_y_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Temporales - Tercera Capa
        self.conv2_z_temporal = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=3,
            padding='same',
            groups=n_feature_maps * 2  # Convolution separable por canal
        )
        self.bn2_z_temporal = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Espaciales - Tercera Capa
        self.conv2_z_spatial = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn2_z_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Camino de Atajo para Bloque 2
        self.shortcut2_temporal = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps,
            kernel_size=1,
            padding='same'
        )
        self.bn_shortcut2_temporal = nn.BatchNorm1d(n_feature_maps)
        
        self.shortcut2_spatial = nn.Conv1d(
            in_channels=n_feature_maps,
            out_channels=n_feature_maps * 2,
            kernel_size=1
        )
        self.bn_shortcut2_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # BLOQUE 3
        # Convoluciones Temporales - Camino Principal
        self.conv3_x_temporal = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=8,
            padding='same',
            groups=n_feature_maps * 2  # Convolution separable por canal
        )
        self.bn3_x_temporal = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Espaciales - Camino Principal
        self.conv3_x_spatial = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn3_x_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Temporales - Segunda Capa
        self.conv3_y_temporal = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=5,
            padding='same',
            groups=n_feature_maps * 2  # Convolution separable por canal
        )
        self.bn3_y_temporal = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Espaciales - Segunda Capa
        self.conv3_y_spatial = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn3_y_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Temporales - Tercera Capa
        self.conv3_z_temporal = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=3,
            padding='same',
            groups=n_feature_maps * 2  # Convolution separable por canal
        )
        self.bn3_z_temporal = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Convoluciones Espaciales - Tercera Capa
        self.conv3_z_spatial = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=n_feature_maps * 2,
            kernel_size=1  # Convolución puntual
        )
        self.bn3_z_spatial = nn.BatchNorm1d(n_feature_maps * 2)
        
        # BatchNorm para atajo en Bloque 3 (sin convolución)
        self.bn_shortcut3 = nn.BatchNorm1d(n_feature_maps * 2)
        
        # Capa de proyección final para regresión (C, T) → (output_channels, T)
        self.output_conv = nn.Conv1d(
            in_channels=n_feature_maps * 2,
            out_channels=output_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        """
        Realiza la pasada hacia adelante del modelo.
        
        Entrada:
          - x: tensor de forma (B, C, T)
               B: tamaño del batch
               C: canales de entrada (input_channels)
               T: pasos temporales (time_steps)
        
        Salida:
          - tensor de forma (B, output_channels, T)
        """
        # Verificar que la entrada tiene el formato correcto (B, C, T)
        if x.dim() != 3:
            raise ValueError(f"La entrada debe tener forma (batch_size, input_channels, time_steps). Shape actual: {x.shape}")
        
        # BLOQUE 1 - Camino Principal
        # Convoluciones Temporales
        conv1_x = self.conv1_x_temporal(x)
        conv1_x = self.bn1_x_temporal(conv1_x)
        conv1_x = F.elu(conv1_x)
        
        # Convoluciones Espaciales
        conv1_x = self.conv1_x_spatial(conv1_x)
        conv1_x = self.bn1_x_spatial(conv1_x)
        conv1_x = F.elu(conv1_x)
        
        # Segunda capa de convoluciones
        conv1_y = self.conv1_y_temporal(conv1_x)
        conv1_y = self.bn1_y_temporal(conv1_y)
        conv1_y = F.elu(conv1_y)
        
        conv1_y = self.conv1_y_spatial(conv1_y)
        conv1_y = self.bn1_y_spatial(conv1_y)
        conv1_y = F.elu(conv1_y)
        
        # Tercera capa de convoluciones
        conv1_z = self.conv1_z_temporal(conv1_y)
        conv1_z = self.bn1_z_temporal(conv1_z)
        conv1_z = F.elu(conv1_z)
        
        conv1_z = self.conv1_z_spatial(conv1_z)
        conv1_z = self.bn1_z_spatial(conv1_z)
        
        # BLOQUE 1 - Camino de Atajo
        shortcut1 = self.shortcut1_temporal(x)
        shortcut1 = self.bn_shortcut1_temporal(shortcut1)
        shortcut1 = F.elu(shortcut1)
        
        shortcut1 = self.shortcut1_spatial(shortcut1)
        shortcut1 = self.bn_shortcut1_spatial(shortcut1)
        
        # BLOQUE 1 - Suma y Activación
        output_block_1 = shortcut1 + conv1_z
        output_block_1 = F.relu(output_block_1)
        
        # BLOQUE 2 - Camino Principal
        # Convoluciones Temporales
        conv2_x = self.conv2_x_temporal(output_block_1)
        conv2_x = self.bn2_x_temporal(conv2_x)
        conv2_x = F.elu(conv2_x)
        
        # Convoluciones Espaciales
        conv2_x = self.conv2_x_spatial(conv2_x)
        conv2_x = self.bn2_x_spatial(conv2_x)
        conv2_x = F.elu(conv2_x)
        
        # Segunda capa de convoluciones
        conv2_y = self.conv2_y_temporal(conv2_x)
        conv2_y = self.bn2_y_temporal(conv2_y)
        conv2_y = F.elu(conv2_y)
        
        conv2_y = self.conv2_y_spatial(conv2_y)
        conv2_y = self.bn2_y_spatial(conv2_y)
        conv2_y = F.elu(conv2_y)
        
        # Tercera capa de convoluciones
        conv2_z = self.conv2_z_temporal(conv2_y)
        conv2_z = self.bn2_z_temporal(conv2_z)
        conv2_z = F.elu(conv2_z)
        
        conv2_z = self.conv2_z_spatial(conv2_z)
        conv2_z = self.bn2_z_spatial(conv2_z)
        
        # BLOQUE 2 - Camino de Atajo
        shortcut2 = self.shortcut2_temporal(output_block_1)
        shortcut2 = self.bn_shortcut2_temporal(shortcut2)
        shortcut2 = F.elu(shortcut2)
        
        shortcut2 = self.shortcut2_spatial(shortcut2)
        shortcut2 = self.bn_shortcut2_spatial(shortcut2)
        
        # BLOQUE 2 - Suma y Activación
        output_block_2 = shortcut2 + conv2_z
        output_block_2 = F.relu(output_block_2)
        
        # BLOQUE 3 - Camino Principal
        # Convoluciones Temporales
        conv3_x = self.conv3_x_temporal(output_block_2)
        conv3_x = self.bn3_x_temporal(conv3_x)
        conv3_x = F.elu(conv3_x)
        
        # Convoluciones Espaciales
        conv3_x = self.conv3_x_spatial(conv3_x)
        conv3_x = self.bn3_x_spatial(conv3_x)
        conv3_x = F.elu(conv3_x)
        
        # Segunda capa de convoluciones
        conv3_y = self.conv3_y_temporal(conv3_x)
        conv3_y = self.bn3_y_temporal(conv3_y)
        conv3_y = F.elu(conv3_y)
        
        conv3_y = self.conv3_y_spatial(conv3_y)
        conv3_y = self.bn3_y_spatial(conv3_y)
        conv3_y = F.elu(conv3_y)
        
        # Tercera capa de convoluciones
        conv3_z = self.conv3_z_temporal(conv3_y)
        conv3_z = self.bn3_z_temporal(conv3_z)
        conv3_z = F.elu(conv3_z)
        
        conv3_z = self.conv3_z_spatial(conv3_z)
        conv3_z = self.bn3_z_spatial(conv3_z)
        
        # BLOQUE 3 - Camino de Atajo (solo BatchNorm)
        shortcut3 = self.bn_shortcut3(output_block_2)
        
        # BLOQUE 3 - Suma y Activación
        output_block_3 = shortcut3 + conv3_z
        output_block_3 = F.relu(output_block_3)
        
        # CAPA FINAL - Proyección a output_channels
        output = self.output_conv(output_block_3)
        
        return output