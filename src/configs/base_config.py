from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    data_path: str
    batch_size: int = 1
    n_splits:int =5 
    patience:int = 10
    optuna_epochs: int = 50
    final_epochs: int = 100
    optuna_trials: int = 5
    variants: Tuple[str, ...] = ("inceptiontime", "disjointcnn", "disjointresnet", "fcn", "multichanneldcnn",
        "dilatedcnn","multiscalecnn", "lite", "resnet", "resnetcnn", "zhaocnn",
        "deepresidualbilstm", "cnnbilstmbigrudil", "cnngru", "cnnlstm", "cnnlstmgru", "cnnbigru", "cnnbilstm", "cnnbilstmbigru"
        )
    verbose: bool = False
    input_channels: int = 12
    output_channels: int = 256
    time_steps: int = 256

