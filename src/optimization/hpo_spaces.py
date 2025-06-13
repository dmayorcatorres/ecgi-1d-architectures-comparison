"""
Define los rangos de búsqueda de manera **agnóstica al backend**
y expone dos funciones de conveniencia:

    sample_optuna(trial, variant)  → dict con valores muestreados
    ray_search_space(variant)      → dict con objetos ray.tune.*

Añadir un nuevo modelo = añadir su entrada
"""
from __future__ import annotations
from typing import Dict, Any, Sequence
import optuna
from ray import tune

# ────────── Rangos reutilizables ──────────
INT_RANGE        = lambda low, high, step: ("int", low, high, step)
FLOAT_RANGE      = lambda low, high, step: ("float", low, high, step)
LOGUNIFORM_RANGE = lambda a, b: ("loguniform", a, b)
CHOICE           = lambda opts: ("choice", opts)

R_INT     = INT_RANGE(32, 192, 32) # Número de filtros
H_INT     = INT_RANGE(16, 128, 16) # Hidden size
SE_INT = INT_RANGE(8 , 32, 8)
TL_INT  = INT_RANGE(8, 64, 8) # Time Layes GRU/LSTM
R_LOG_LR  = CHOICE([3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2])

NL_INT    = INT_RANGE(1, 3, 1) # Number of layers
DROPOUT_FLOAT = FLOAT_RANGE(0.2, 0.5, 0.1) # Dropout rate
R_CHOICE_BOOL       = CHOICE([True, False])
KERNEL_SIZE_INT = INT_RANGE(3, 9, 2) # Kernel size

# ───────────────── diccionario maestro ─────────────────
# Cada hiper-parámetro se describe con una tupla:
#   (tipo,  *valores)
# tipos soportados: "choice", "int", "float", "loguniform"
SPACE: Dict[str, Dict[str, tuple]] = {
        # ──────────────────────── CNNs ────────────────────────
    "dilatedcnn": {
    "stage1_out": TL_INT,
    "fc_units": R_INT,
    "lr": R_LOG_LR,
    },

    "disjointcnn": {
    "stage1_out": R_INT,
    "lr": R_LOG_LR,
    },

    # -------- DenseNet-BiLSTM -------------------------------------------
    "densenetbilstm": {
    "initial_filters": ("int", 16, 64, 16),
    "growth_rate_small": ("int", 16, 64, 16),
    "growth_rate_large": ("int", 32, 128, 32),
    "num_layers_small": ("int", 2, 4, 1),
    "num_layers_large": ("int", 1, 3, 1),
    "transition_small": ("int", 16, 64, 16),
    "transition_large": ("int", 32, 128, 32),
    "lr": ("loguniform", 1e-4, 1e-2),
    },

    "disjointresnet": {
    "stage1_out": R_INT,
    "lr": R_LOG_LR,
    },

    "fcn": {
    "stage1_out": R_INT,
    "lr": R_LOG_LR,
    },

    "inceptiontime": {
    "stage1_out": R_INT,
    "bottleneck_channels": TL_INT,
    "use_residual": ("choice", [True, False]),
    "num_inception_modules": KERNEL_SIZE_INT,
    "lr": R_LOG_LR,
    },

    "lite": {
    "stage1_out": R_INT,
    "base_kernel_size": ("int", 20, 60, 20),
    "use_custom_filters":("choice", [True, False]),
    "lr": R_LOG_LR,
    },

    "mlp": {
    "stage1_out": R_INT,
    "stage2_out": R_INT,
    "stage3_out": R_INT,
    "dropout_rate": DROPOUT_FLOAT,
    "lr": R_LOG_LR,
    },

    "multichanneldcnn": {
    "sub_channels": TL_INT,
    "lr": R_LOG_LR,
    },

    "multiscalecnn": {
        "stage1_out": R_INT,
        "stage2_out": R_INT,
        "stage3_out": R_INT,

        # arquitectura
        "kernel_size": ("choice", [(3, 5, 7), (5, 7, 9), (7, 9, 11)]),
        "lr": R_LOG_LR,

    },

    "oscnn": {
        # ---------- Bloque Omni-Scale ----------
        "base_channels": ("int", 16, 128),
        "max_rf":        ("int", 32, 512),
        "max_primes":    ("choice", [8, 12, 16, 24, 32], None),
        "bottleneck":    ("choice", [4, 8], None),
        "dropout":       ("float", 0.0, 0.5),

        # ---------- Optimización ----------
        "lr":            ("log_float", 1e-4, 1e-2),
        "batch_size":    ("int", 8, 64),
    },


    "resnet": {
    "stage1_out": R_INT,
    "stage2_out": R_INT,
    "stage3_out": R_INT,
    "lr": R_LOG_LR,
    },

    "resnetcnn": {
        "stage1_out": R_INT,
        "kernel_size_in": KERNEL_SIZE_INT,
        "lr":          R_LOG_LR,
    },

    "tfrcnn": {
        "stage1_out": R_INT,
        "stage2_out": R_INT,
        "stage3_out": R_INT,
        "lr":         R_LOG_LR,
    },

    "zhaocnn": {
        "stage1_out": R_INT,
        "stage2_out": R_INT,
        "stage3_out": R_INT,
        "lr":         R_LOG_LR,
    },
        "tcn": {
        "n_blocks":   ("int", 4, 8, 1),
        "n_filters":  R_INT,
        "kernel_size": KERNEL_SIZE_INT,
        "dropout":    DROPOUT_FLOAT,
        "lr":         R_LOG_LR,
    },
    
    # ──────────────────────── RNN “puros” ────────────────────────
    "rnn_gru": {
        "gru_hidden":   H_INT,
        "num_layers":   NL_INT,
        "bidirectional":R_CHOICE_BOOL,
        "dropout_rate": DROPOUT_FLOAT,
        "lr":           R_LOG_LR,
    },

    "rnn_lstm": {
        "lstm_hidden":  H_INT,
        "num_layers":   NL_INT,
        "bidirectional":R_CHOICE_BOOL,
        "dropout_rate": DROPOUT_FLOAT,
        "lr":           R_LOG_LR,
    },

    "peepholelstm": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "bidirectional": R_CHOICE_BOOL,     # fijo
        "lr":          R_LOG_LR,
    },

    "phasedlstm": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "period_init": ("float", 0.5, 5, 0.5),
        "r_on_init":   ("float", 0.01, 0.2, 0.01),
        "lr":          R_LOG_LR,
    },

    "indrnn": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "dropout_rate": DROPOUT_FLOAT,
        "clip_max":    ("float", 0.5, 2.0, 0.5),
        "lr":          R_LOG_LR,
    },

    "qrnn": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "kernel_size": KERNEL_SIZE_INT,
        "dropout_rate": DROPOUT_FLOAT,
        "pool_type":   ("choice", ["fo", "f", "o"]),
        "lr":          R_LOG_LR,
    },

    "esn": {
        "reservoir_size": ("int", 100, 1000, 100),
        "spectral_radius":("float", 0.1, 1.5, 0.1),
        "sparsity":       ("float", 0.0, 0.9, 0.1),
        "ridge_alpha":    ("loguniform", 1e-6, 1e-1),
        "lr":             R_LOG_LR,
    },

    "tlgru": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "dropout":     DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },

    "vsgrunet": {
        "hidden_size": R_INT,
        "num_layers":  NL_INT,
        "dropout":     DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },


    # ───────────────────────── CNN-RNN híbridos (falta) ─────────────────────────

    "deepresidualbilstm": {
        "hidden_dim": R_INT,
        "num_layers": ("int", 1, 5, 1),
        "dropout": DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },

    "cnngru": {
        "stage1_out":   R_INT,
        "stage2_out":   R_INT,
        "stage3_out":   R_INT,
        "stage4_out":  R_INT,
        "rnn_type":    ("choice", ["GRU", "LSTM"]),
        "bidirectional": R_CHOICE_BOOL,           # fijo
        "dropout_rate": DROPOUT_FLOAT,
        "dropout_rate": DROPOUT_FLOAT,
        "se_ratio":   SE_INT, 
        "use_se":      R_CHOICE_BOOL,
        "lr":          R_LOG_LR,
    },
    
    "cnnlstm": {
        "stage1_out":  R_INT,
        "stage2_out":  R_INT,
        "stage3_out":  R_INT,
        "stage4_out":  R_INT,
        "lstm_hidden": TL_INT,
        "bidirectional": R_CHOICE_BOOL,           # fijo
        "dropout_rate": DROPOUT_FLOAT,
        "se_ratio":    SE_INT,
        "use_se":      R_CHOICE_BOOL,
        "lr":          R_LOG_LR,
    },

    "cnnlstmgru": {
        "stage1_out":  R_INT,
        "stage2_out":  R_INT,
        "stage3_out":  R_INT,
        "stage4_out":  R_INT,
        "lstm_hidden": TL_INT,
        "gru_hidden": TL_INT,
        "lstm_bidirectional": R_CHOICE_BOOL,
        "gru_bidirectional":  R_CHOICE_BOOL,
        "dropout_rate": DROPOUT_FLOAT,
        "se_ratio":    SE_INT,
        "use_se":      R_CHOICE_BOOL,
        "lr":          R_LOG_LR,
    },
    "disjointlstm": {
        "residual": R_CHOICE_BOOL,
        "bidirectional": R_CHOICE_BOOL, 
        "hidden_size": R_INT,
        "lr":          R_LOG_LR,
    },

    "lstmfcn": {
        "conv1_out":   R_INT,
        "conv2_out":  R_INT,
        "conv3_out":   R_INT,
        "lstm_hidden": R_INT,
        "bidirectional": R_CHOICE_BOOL,
        "dropout_rate": DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },

    "cnnbilstmbigrudil": {
        "cnn_channels": R_INT,
        "gru_hidden":   TL_INT,
        "lstm_hidden":  TL_INT,
        "dropout": DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },

    # Attention based
    "alstm": {
        "stage1_out": R_INT,
        "stage2_out": R_INT,
        "stage3_out": R_INT,
        "lstm_hidden":TL_INT,
        "lr":          R_LOG_LR,
    },
    "transformer1d": {
        "d_model":     R_INT,  # Dimensión del modelo
        "num_heads":   ("choice", [2, 4, 8]),  # Número de cabezas de atención
        "num_layers":  ("choice", [1, 2, 3, 4, 5, 6]),  # Número de capas del codificador
        "dropout":     DROPOUT_FLOAT,          # Tasa de dropout
        "lr":          R_LOG_LR,               # Tasa de aprendizaje
    },
    "informer1d": {
        "d_model":     R_INT,  # Dimensión del modelo
        "num_heads":   ("choice", [2, 4, 8]),  # Número de cabezas de atención
        "num_layers":  ("choice", [1, 2, 3, 4, 5, 6]),  # Número de capas del codificador
        "dropout":     DROPOUT_FLOAT,          # Tasa de dropout
        "lr":          R_LOG_LR,               # Tasa de aprendizaje
    },
    "autoformer1d": {
        "d_model":     R_INT,  # Dimensión del espacio latente
        "num_heads":   ("choice", [2, 4, 8]),            # Número de cabezas de atención
        "num_layers":  ("choice", [1, 2, 3, 4, 6]),       # Número de capas del codificador
        "dropout":     DROPOUT_FLOAT,            # Tasa de dropout
        "lr":         R_LOG_LR,       # Tasa de aprendizaje
    },

    "patchtst": {
        "patch_len":    ("choice", [8, 16, 32]),     # Tamaño del patch temporal
        "stride":       ("choice", [4, 8, 16]),      # Paso entre patches
        "d_model":      R_INT,                      # Dimensión del modelo latente
        "num_heads":    ("choice", [2, 4, 8]),       # Número de cabezas de atención
        "num_layers":   ("choice", [1, 2, 3, 4, 5, 6]),  # Capas del codificador
        "dropout":      DROPOUT_FLOAT,              # Dropout en las capas
        "lr":           R_LOG_LR,                   # Tasa de aprendizaje
    },

    "vit1d": {
        "d_model":     R_INT,                        # Dimensión del modelo
        "num_heads":   ("choice", [2, 4, 8]),        # Número de cabezas de atención
        "num_layers":  ("choice", [2, 4, 6]),        # Capas Transformer
        "mlp_dim":     ("choice", [128, 256, 512]),  # Dimensión del feedforward interno
        "dropout":     DROPOUT_FLOAT,                # Tasa de dropout
        "lr":          R_LOG_LR,                     # Tasa de aprendizaje
        "patch_size":  ("choice", [4, 8, 16, 17]),  # Tamaño de patch (debe dividir a T)
    },


    "timenet1d": {
        "hidden_size":   R_INT,  # Tamaño oculto GRU
        "num_layers":    ("int", 1, 6),         # Número de capas GRU
        "dropout":       DROPOUT_FLOAT,  # Dropout
        "lr":            R_LOG_LR, # Learning rate
    },

    "bertbigru": {
        "hidden_dim":  R_INT,
        "num_layers":  ("choice", [1, 2, 3]),
        "dropout_rate": DROPOUT_FLOAT,
        "lr":          R_LOG_LR,
    },


}

# … resto del fichero idéntico …

# ───────── helper Optuna ─────────
def sample_optuna(trial: optuna.Trial, variant: str) -> Dict[str, Any]:
    if variant not in SPACE:
        raise ValueError(f"Variante «{variant}» sin espacio definido")

    return {hp: _optuna_suggest(trial, hp, spec)
            for hp, spec in SPACE[variant].items()}


# ───────── helper Ray ─────────
def _ray_translate(spec: tuple):
    kind, *vals = spec
    if kind == "choice":
        return tune.choice(vals[0])

    if kind == "int":
        lo, hi, step = vals
        return tune.choice(list(range(lo, hi + step, step)))   # mantiene ‘step’

    if kind == "float":
        lo, hi, step = vals
        return tune.quniform(lo, hi, step)                     # respeta paso

    if kind == "loguniform":
        lo, hi = vals
        return tune.loguniform(lo, hi)

    raise ValueError(kind)


def ray_search_space(variant: str) -> Dict[str, Any]:
    if variant not in SPACE:
        raise ValueError(f"Variante «{variant}» sin espacio definido")
    return {hp: _ray_translate(spec) for hp, spec in SPACE[variant].items()}
