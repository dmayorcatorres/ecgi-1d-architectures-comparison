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
    
    #variants: Tuple[str, ...] = ("inceptiontime", "disjointcnn", "disjointresnet", "fcn", "multichanneldcnn", "mlp",
    #    "dilatedcnn","multiscalecnn", "lite", "resnet", "resnetcnn", "zhaocnn",
    #    "deepresidualbilstm", "cnnbilstmbigrudil", "cnngru", "cnnlstm", "cnnlstmgru", "cnnbigru", "cnnbilstm", "cnnbilstmbigru"
    #    )
    
    # CNNs
    #variants: Tuple[str, ...] = ("disjointcnn", "disjointresnet", "fcn", "inceptiontime", "multichanneldcnn",
    #    "multiscalecnn", "dilatedcnn", "lite", "resnet", "resnetcnn", "tfrcnn", "zhaocnn"
    #    )

    variants: Tuple[str, ...] = ("vsgrunet", ) # tcn, "tlgru"

    # RNNs
    #variants: Tuple[str, ...] = ("cnngru", "cnnlstm", "cnnbigru", "cnnbilstm","cnnlstmgru",  "cnnbilstmbigru, "deepresidualbilstm", "cnnbilstmbigrudil","
    #    )

    #variants: Tuple[str, ...] = ("mlp",)
    
    verbose: bool = False
    input_channels: int = 12
    output_channels: int = 256
    time_steps: int = 256

