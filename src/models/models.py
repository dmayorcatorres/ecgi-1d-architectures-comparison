import torch
import torch.nn as nn
from typing import Type

# Basados en CNN
from src.models.DisjointCNN import DisjointCNN
from src.models.DisjointResNet import DisjointResNet
from src.models.FCN import FCN
from src.models.InceptionTime import InceptionTime 
from src.models.MultiChannelDCNN import MultiChannelDCNN
from src.models.MultiScaleCNN import MultiScaleCNN
from src.models.DilatedCNN import DilatedCNN
from src.models.LITE import LITE
from src.models.ResNet import ResNet
from src.models.ResNetCNN import ResNetCNN
from src.models.ZhaoCNN import ZhaoCNN  # Ajuste el nombre de la clase si es diferente
 

from src.models.CNN_LSTM import CNN_LSTM
from src.models.CNN_GRU import CNN_GRU
from src.models.CNN_LSTM_GRU import CNN_LSTM_GRU
from src.models.CNN_BILSTM import CNN_BILSTM
from src.models.CNN_BIGRU import CNN_BIGRU
from src.models.CNN_BILSTM_BIGRU import CNN_BILSTM_BIGRU
from src.models.CNN_BILSTM_BIGRU_Dilated import CNN_BILSTM_BIGRU_Dilated

from src.models.CNN_ALSTM import CNN_ALSTM
#from src.models.DisjointLSTM import DisjointLSTM

#from src.models.BERT_BIGRU import BERT_BiGRU
from src.models.DeepResidualBiLSTM import DeepResidualBiLSTM



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
            # CNNs
            "disjointcnn": DisjointCNN,
            "disjointresnet": DisjointResNet,
            "fcn": FCN,
            "inceptiontime": InceptionTime,
            "multichanneldcnn": MultiChannelDCNN,
            "multiscalecnn": MultiScaleCNN,
            "dilatedcnn": DilatedCNN,
            "lite": LITE,
            "resnet": ResNet,
            "resnetcnn": ResNetCNN,
            "zhaocnn": ZhaoCNN,
            # RNNs
            "cnngru": CNN_GRU,
            "cnnlstm": CNN_LSTM,
            "cnnlstmgru": CNN_LSTM_GRU,
            "cnnbilstm": CNN_BILSTM,
            "cnnbigru": CNN_BIGRU,
            "cnnbilstmbigru": CNN_BILSTM_BIGRU,
            "cnnbilstmbigrudil": CNN_BILSTM_BIGRU_Dilated,
            "deepresidualbilstm": DeepResidualBiLSTM,


            "disjointlstm": DisjointLSTM,

            "cnnalstm": CNN_ALSTM,
#            "bertbigru": BERT_BiGRU,

        }


        if self.variant in model_dict:
            self.model = model_dict[self.variant](time_steps, input_channels, output_channels, **kwargs)
        else:
            raise ValueError(f"Variante desconocida: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
