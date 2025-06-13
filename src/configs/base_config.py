# src/configs/base_config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    # ─── Obligatorios (sin default) ──────────────────────────────────────
    data_path: str                       # 1º: va antes de cualquier campo con default

    # ─── HPO ─────────────────────────────────────────────────────────────
    hpo_backend: str = "ray"             # "ray" | "optuna"
    n_trials:    int  = 30              # nº de muestras por variante

    # ─── Entrenamiento / CV ─────────────────────────────────────────────
    batch_size:     int = 1
    n_splits:       int = 5
    final_epochs:   int = 200

    
    optuna_epochs:  int = 5
    optuna_trials:  int = 5
    
    # ─── Otros parámetros ───────────────────────────────────────────────
    verbose:         bool = False
    input_channels:  int  = 12
    output_channels: int  = 256
    time_steps:      int  = 256


    # ─── Variantes a evaluar ────────────────────────────────────────────

    # Hybrids
    #variants: Tuple[str, ...] = ("deepresidualbilstm", "cnngru", "cnnlstm", "cnnlstmgru", "disjointlstm", "lstmfcn", "cnnbilstmbigrudil")
    
    # CNNs
    variants: Tuple[str, ...] = ("dilatedcnn", "disjointcnn", "disjointresnet", "fcn", #"inceptiontime", "lite", "mlp",  "multichanneldcnn", "multiscalecnn", #"oscnn" ,
        "resnet", "resnetcnn", "tcn", "tfrcnn", "zhaocnn",)

    # RNNs
    #variants: Tuple[str, ...] = ("esn", "indrnn","peepholelstm" , "qrnn", "tlgru", "vsgrunet", "rnn_gru", "rnn_lstm",)

    # Transformers / Attentions
    #variants: Tuple[str, ...] = ("alstm", "transformer1d", "autoformer1d", "informer1d", "patchtst",) # "atbased", "transformer1d", "autoformer1d", "informer1d", "patchtst", "vit1d")

