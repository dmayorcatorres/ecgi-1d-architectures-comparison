import torch.nn as nn
import optuna
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
from src.models.DeepResidualBiLSTM import DeepResidualBiLSTM

from src.models.CNN_ALSTM import CNN_ALSTM

class ModelFactory:
    def __init__(self, config):
        self.config = config
    
    def create_model(self, variant: str, params: dict) -> nn.Module:
        model_class = self._get_model_class(variant)
        return model_class(
            time_steps=self.config.time_steps,
            input_channels=self.config.input_channels,
            output_channels=self.config.output_channels,
            **params
        )
    
    def _get_model_class(self, variant: str):
        # Mapeo de nombres de variantes a clases de modelo
        return {
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
        }[variant]

class HyperparameterOptimizer:
    @staticmethod
    def sample_parameters(trial: optuna.Trial, variant: str) -> dict:
        return hyperparameter_space[variant](trial)