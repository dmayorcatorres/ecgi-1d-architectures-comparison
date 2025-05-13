import torch.nn as nn
import optuna

# CNN based Models
from src.models.cnnbased.DilatedCNN import DilatedCNN
from src.models.cnnbased.DisjointCNN import DisjointCNN
from src.models.cnnbased.DisjointResNet import DisjointResNet
from src.models.cnnbased.FCN import FCN
from src.models.cnnbased.InceptionTime import InceptionTime 
from src.models.cnnbased.LITE import LITE
from src.models.cnnbased.MLP import MLP
from src.models.cnnbased.MultiChannelDCNN import MultiChannelDCNN
from src.models.cnnbased.MultiScaleCNN import MultiScaleCNN
from src.models.cnnbased.OSCNN import OSCNN
from src.models.cnnbased.ResNet import ResNet
from src.models.cnnbased.ResNetCNN import ResNetCNN
from src.models.cnnbased.TCN1D import TCN1D
from src.models.cnnbased.TFRCNN import TFRCNN
from src.models.cnnbased.ZhaoCNN import ZhaoCNN
#from src.models.DenseNetBiLSTM import DenseNetBiLSTM
#from src.models.EfficientNet1D import EfficientNet1D

# RNN based Models
from src.models.rnnbased.TLGRUNet import TLGRUNet
from src.models.rnnbased.VSGRUNet import VSGRUNet
from src.models.DeepResidualBiLSTM import DeepResidualBiLSTM
from src.models.rnnbased.RNN_GRU import RNN_GRU
from src.models.rnnbased.RNN_LSTM import RNN_LSTM
from src.models.rnnbased.PeepholeLSTM import PeepholeLSTM
from src.models.rnnbased.PhasedLSTM import PhasedLSTM
from src.models.rnnbased.IndRNN import IndRNN
from src.models.rnnbased.ESN import ESN
from src.models.rnnbased.QRNN import QRNN


# Hybrids based Models
from src.models.hybridbased.CNN_LSTM import CNN_LSTM
from src.models.hybridbased.CNN_GRU import CNN_GRU
from src.models.hybridbased.CNN_LSTM_GRU import CNN_LSTM_GRU
from src.models.hybridbased.LSTM_FCN import LSTM_FCN
from src.models.hybridbased.CNN_BILSTM_BIGRU_Dilated import CNN_BILSTM_BIGRU_Dilated
from src.models.hybridbased.DisjointLSTM import DisjointLSTM

# Attention based models
from src.models.atbased.CNN_ALSTM import CNN_ALSTM

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
            # CNN-based
            "dilatedcnn": DilatedCNN,
            "disjointcnn": DisjointCNN,
            "disjointresnet": DisjointResNet,
            "fcn": FCN,
            "inceptiontime": InceptionTime,
            "lite": LITE,
            "mlp": MLP,
            "multichanneldcnn": MultiChannelDCNN,
            "multiscalecnn": MultiScaleCNN,
            "oscnn": OSCNN,
            "resnet": ResNet,
            "resnetcnn": ResNetCNN,
            "tcn": TCN1D,
            "tfrcnn": TFRCNN,
            "zhaocnn": ZhaoCNN,
            #"densenetbilstm": DenseNetBiLSTM,
            #"efficientnet1d": EfficientNet1D,

            # RNN-based
            "rnn_gru": RNN_GRU,
            "rnn_lstm": RNN_LSTM,
            "peepholelstm": PeepholeLSTM,
            "phasedlstm": PhasedLSTM,
            "indrnn": IndRNN,
            "qrnn":  QRNN,
            "esn":   ESN,
            "deepresidualbilstm": DeepResidualBiLSTM,
            "tlgru": TLGRUNet,
            "vsgrunet": VSGRUNet,
            
            # Hybrid
            "cnnbilstmbigrudil": CNN_BILSTM_BIGRU_Dilated,
            "cnngru": CNN_GRU,
            "cnnlstmgru": CNN_LSTM_GRU,
            "cnnlstm": CNN_LSTM,
            "disjointlstm": DisjointLSTM,
            "lstm_fcn": LSTM_FCN,
            



        }[variant]

class HyperparameterOptimizer:
    @staticmethod
    def sample_parameters(trial: optuna.Trial, variant: str) -> dict:
        return hyperparameter_space[variant](trial)