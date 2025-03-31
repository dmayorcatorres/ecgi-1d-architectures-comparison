import torch
import torch.nn as nn
from typing import Type

# Importar los modelos (asegúrese de que los nombres de las clases importadas sean correctos)
from src.models.ResNet import ResNet
from src.models.DilatedCNN import DilatedCNN
from src.models.DisjointCNN import DisjointCNN
from src.models.FCN import FCN
from src.models.MultiScaleCNN import MultiScaleCNN
from src.models.MultiChannelDCNN import MultiChannelDCNN
from src.models.ZhaoCNN import ZhaoCNN  # Ajuste el nombre de la clase si es diferente
from src.models.ResNetCNN import ResNetCNN
from src.models.InceptionTime import InceptionTime  
from src.models.CNN_LSTM import CNN_LSTM
from src.models.DisjointLSTM import DisjointLSTM
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
            "dilatedcnn": DilatedCNN,
            "disjointcnn": DisjointCNN,
            "disjointlstm": DisjointLSTM,
            "disjointresnet": DisjointResNet,
            "fcn": FCN,
            "inceptiontime": InceptionTime,
            "lite": LITE,
            "cnnlstm": CNN_LSTM,
            "multichanneldcnn": MultiChannelDCNN,
            "multiscalecnn": MultiScaleCNN,
            "resnet": ResNet,
            "resnetcnn": ResNetCNN,
            "zhaocnn": ZhaoCNN
        }


        if self.variant in model_dict:
            self.model = model_dict[self.variant](time_steps, input_channels, output_channels, **kwargs)
        else:
            raise ValueError(f"Variante desconocida: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
