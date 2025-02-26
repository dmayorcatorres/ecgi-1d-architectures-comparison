import torch
import torch.nn as nn
from typing import Type

# Importar los modelos (asegúrese de que los nombres de las clases importadas sean correctos)
from src.models.ResNet import ResNet
from src.models.Dilated2DCNN import Dilated2DCNN
from src.models.Disjoint2DCNN import Disjoint2DCNN
from src.models.FCN import FCN
from src.models.MultiScaleCNN import MultiScaleCNN
from src.models.MultiChannelDCNN import MultiChannelDCNN
from src.models.MCCNN import MCCNN
from src.models.ZHMCCNN import ZHMCCNN  # Ajuste el nombre de la clase si es diferente
from src.models.ResNetCNN import ResNetCNN
from src.models.InceptionTime import InceptionTime  
from src.models.LSTM_FCN import LSTM_FCN
from src.models.LSTMDisjoint import LSTMDisjoint
from src.models.DisjointResNet import DisjointResNet
from src.models.LITE import LITE


class RawCNNBase(nn.Module):
    """
    RawCNNBase – Clase base para modelos CNN 1D para clasificación de series temporales.
    
    Según el parámetro `variant`, se instancia de manera dinámica el modelo correspondiente.
    
    Variantes disponibles:
      - "fcn"       : FCNModel
      - "resnet"    : ResNetCNNModel
      - "dilated"   : Dilated2DCNN
      - "disjoint"  : Disjoint2DCNN
      - "multiscale": MultiScaleCNN
      - "mc-dcnn"   : MC_DCNN_Model
      - "mc-cnn"    : MCCNNModel
      - "zhmc-cnn"  : ZHMCCNN
      - "res-cnn"   : ResCNNIntegrationModel
      
    Parámetros:
      - variant (str): Variante del modelo a utilizar.
      - time_steps (int): Número de pasos temporales.
      - input_channels (int): Número de canales de entrada.
      - output_channels (int): Número de canales de salida o clases.
      - **kwargs: Otros parámetros que se pasan al constructor del modelo seleccionado.
    """
    
    def __init__(self, variant: str, time_steps: int, input_channels: int, output_channels: int, **kwargs) -> None:
        super().__init__()
        self.variant = variant.lower()
        self.time_steps = time_steps
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Diccionario que asocia la variante con la clase correspondiente.
        model_dict: dict[str, Type[nn.Module]] = {
            "fcn": FCN,
            "resnet": ResNet,  # Aquí se asume que ResNetCNNModel es importado como ResNetCNN
            "dilated": Dilated2DCNN,
            "disjoint": Disjoint2DCNN,
            "mc-dcnn": MultiChannelDCNN,
            "mc-cnn": MCCNN,
            "zhmc-cnn": ZHMCCNN,
            "res-cnn": ResNetCNN,
            "inception": InceptionTime,
            "lstm-fcn": LSTM_FCN,
            "lstm-disjoint": LSTMDisjoint,
            "disjoint-resnet": DisjointResNet,
            "lite": LITE
        }

        if self.variant in model_dict:
            self.model = model_dict[self.variant](time_steps, input_channels, output_channels, **kwargs)
        else:
            raise ValueError(f"Variante desconocida: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
