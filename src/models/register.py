"""
models/model_registry.py
─────────────────────────────────────────────────────────
Diccionarios globales para:

    • MODEL_REGISTRY : «alias» → clase nn.Module
    • SPACE_REGISTRY : «alias» → dict  o  función que devuelve dict

Se registran de forma **explícita** con `register_model`
y `@register_space(alias)`.  
Añade o quita modelos/espacios editando únicamente este archivo.
"""

from typing import Dict, Type, Callable, Union
import torch.nn as nn

# ================================================================
# 1. Diccionarios globales
# ================================================================
MODEL_REGISTRY: Dict[str, Type[nn.Module]]                = {}
SPACE_REGISTRY: Dict[str, Union[dict, Callable[[], dict]]] = {}

# ================================================================
# 2. Funciones helper
# ================================================================
def register_model(name: str, cls: Type[nn.Module]):
    """Registra la clase del modelo."""
    key = name.lower()
    if key in MODEL_REGISTRY:
        raise KeyError(f"Modelo '{key}' ya registrado")
    MODEL_REGISTRY[key] = cls


def register_space(name: str):
    """Devuelve un decorador para asociar el search-space a «name»."""
    key = name.lower()

    def _wrapper(obj):
        SPACE_REGISTRY[key] = obj          # dict o función
        return obj                         # permite encadenar decoradores
    return _wrapper

# ================================================================
# 3. IMPORTS explícitos  (+ registro)
#    → Usa imports *relativos* dentro del paquete «models»
# ================================================================
# ---------- CNN based ------------------------------------------------
from .cnnbased.DilatedCNN        import DilatedCNN
from .cnnbased.DisjointCNN       import DisjointCNN
from .cnnbased.DisjointResNet    import DisjointResNet
from .cnnbased.FCN               import FCN
from .cnnbased.InceptionTime     import InceptionTime
from .cnnbased.LITE              import LITE
from .cnnbased.MLP               import MLP
from .cnnbased.MultiChannelDCNN  import MultiChannelDCNN
from .cnnbased.MultiScaleCNN     import MultiScaleCNN
from .cnnbased.ResNet            import ResNet
from .cnnbased.ResNetCNN         import ResNetCNN
from .cnnbased.TCN1D             import TCN1D
from .cnnbased.TFRCNN            import TFRCNN
from .cnnbased.ZhaoCNN           import ZhaoCNN

# ---------- RNN based -------------------------------------------------
from .rnnbased.TLGRUNet          import TLGRUNet
from .rnnbased.VSGRUNet          import VSGRUNet
from .rnnbased.RNN_GRU           import RNN_GRU
from .rnnbased.RNN_LSTM          import RNN_LSTM
from .rnnbased.PeepholeLSTM      import PeepholeLSTM
from .rnnbased.IndRNN            import IndRNN
from .rnnbased.ESN               import ESN
from .rnnbased.QRNN              import QRNN

# ---------- Hybrid ----------------------------------------------------
from .hybridbased.CNN_LSTM                   import CNN_LSTM
from .hybridbased.CNN_GRU                    import CNN_GRU
from .hybridbased.CNN_LSTM_GRU               import CNN_LSTM_GRU
from .hybridbased.LSTM_FCN                   import LSTM_FCN
from .hybridbased.CNN_BILSTM_BIGRU_Dilated   import CNN_BILSTM_BIGRU_Dilated
from .hybridbased.DeepResidualBiLSTM         import DeepResidualBiLSTM
from .hybridbased.DisjointLSTM               import DisjointLSTM

# ---------- Attention -------------------------------------------------
from .atbased.ALSTM           import ALSTM
from .atbased.Transformer1D   import Transformer1D
from .atbased.Informer1D      import Informer1D
from .atbased.Autoformer1D    import Autoformer1D
from .atbased.PatchTST        import PatchTST
from .atbased.ViT1D           import ViT1D
from .atbased.TimeNet1D       import TimeNet1D
from .atbased.PhysMixFormer1D import PhysMixFormer1D

# ----------------- Registro de modelos -------------------------------
register_model("dilatedcnn", DilatedCNN)
register_model("disjointcnn", DisjointCNN)
register_model("disjointresnet", DisjointResNet)
register_model("fcn", FCN)
register_model("inceptiontime", InceptionTime)
register_model("lite", LITE)
register_model("mlp", MLP)
register_model("multichanneldcnn", MultiChannelDCNN)
register_model("multiscalecnn", MultiScaleCNN)
register_model("resnet", ResNet)
register_model("resnetcnn", ResNetCNN)
register_model("tcn", TCN1D)
register_model("tfrcnn", TFRCNN)
register_model("zhaocnn", ZhaoCNN)

register_model("tlgru", TLGRUNet)
register_model("vsgrunet", VSGRUNet)
register_model("rnn_gru", RNN_GRU)
register_model("rnn_lstm", RNN_LSTM)
register_model("peepholelstm", PeepholeLSTM)
register_model("indrnn", IndRNN)
register_model("esn", ESN)
register_model("qrnn", QRNN)

register_model("deepresidualbilstm", DeepResidualBiLSTM)
register_model("cnnlstm", CNN_LSTM)
register_model("cnngru", CNN_GRU)
register_model("cnnlstmgru", CNN_LSTM_GRU)
register_model("lstmfcn", LSTM_FCN)
register_model("disjointlstm", DisjointLSTM)
register_model("cnnbilstmbigrudil", CNN_BILSTM_BIGRU_Dilated)

register_model("alstm", ALSTM)
register_model("transformer1d", Transformer1D)
register_model("informer1d", Informer1D)
register_model("autoformer1d", Autoformer1D)
register_model("patchtst", PatchTST)
register_model("vit1d", ViT1D)
register_model("timenet1d", TimeNet1D)
register_model("physformer", PhysMixFormer1D)

# ----------------- Ejemplo de space (decorador) -----------------------
@register_space("mlp")
def _space_mlp():
    """Search-space para el MLP básico."""
    return {
        "stage1_out":  ("int",   64, 256, 64),
        "stage2_out":  ("int",   32, 128, 32),
        "stage3_out":  ("int",   16,  64, 16),
        "dropout_rate":("float", 0.0, 0.4, 0.1)
    }

# ────────────────────────────────────────────────────────────────
# 4. Espacios HPO (decoradores) para Ray/Optuna, uno por variante
# ────────────────────────────────────────────────────────────────
@register_space("dilatedcnn")
def _space_dilatedcnn():
    return {
        "stage1_out":        ("int",   32, 192, 32),
        "fc_units":          ("int",   32, 192, 32),
    }

@register_space("disjointcnn")
def _space_disjointcnn():
    return {
        "stage1_out": ("int", 32, 192, 32),
    }

@register_space("densenetbilstm")
def _space_densenetbilstm():
    return {
        "initial_filters":   ("int", 16, 64, 16),
        "growth_rate_small": ("int", 16, 64, 16),
        "growth_rate_large": ("int", 32,128, 32),
        "num_layers_small":  ("int",  2,  4,  1),
        "num_layers_large":  ("int",  1,  3,  1),
        "transition_small":  ("int", 16, 64, 16),
        "transition_large":  ("int", 32,128, 32),
    }

# ... sigue para cada variante según tu diccionario maestro ...
# Ejemplo:
@register_space("fcn")
def _space_fcn():
    return {
        "stage1_out": ("int", 32, 192, 32),
    }

@register_space("inceptiontime")
def _space_inceptiontime():
    return {
        "stage1_out":           ("int", 32, 192, 32),
        "bottleneck_channels":  ("int",  8,  64,  8),
        "use_residual":         ("choice", [True, False]),
        "num_inception_modules":("int",  1,   5,  1),
    }

@register_space("lite")
def _space_lite():
    return {
        "stage1_out":        ("int", 32, 192, 32),
        "base_kernel_size":  ("int", 20,  60, 20),
        "use_custom_filters":("choice", [True, False]),
    }


@register_space("multichanneldcnn")
def _space_multichanneldcnn():
    return {
        "sub_channels": ("int",  8,  64, 8),
    }

@register_space("multiscalecnn")
def _space_multiscalecnn():
    return {
        "stage1_out": ("int", 32,192,32),
        "stage2_out": ("int", 32,192,32),
        "stage3_out": ("int", 32,192,32),
        "kernel_size":("choice", [(3,5,7),(5,7,9),(7,9,11)]),
    }
# ────────────────────────────────────────────────────────────────
# Continuación de espacios HPO
# ────────────────────────────────────────────────────────────────

@register_space("oscnn")
def _space_oscnn():
    return {
        # Bloque Omni-Scale
        "base_channels": ("int",    16, 128, 16),
        "max_rf":        ("int",    32, 512, 32),
        "max_primes":    ("choice",[8,12,16,24,32]),
        "bottleneck":    ("choice",[4,8]),
        "dropout":       ("float",  0.0, 0.4, 0.1),
    }

@register_space("resnet")
def _space_resnet():
    return {
        "stage1_out":   ("int", 32, 192, 32),
        "stage2_out":   ("int", 32, 192, 32),
        "stage3_out":   ("int", 32, 192, 32),
    }

@register_space("disjointresnet")
def _space_disjointresnet():
    return {
        "stage1_out": ("int", 32, 192, 32),
    }


@register_space("resnetcnn")
def _space_resnetcnn():
    return {
        "stage1_out":    ("int", 32, 192, 32),
        "kernel_size_in":("int", 3,   9,   2),
    }

@register_space("tfrcnn")
def _space_tfrcnn():
    return {
        "stage1_out": ("int", 32, 192, 32),
        "stage2_out": ("int", 32, 192, 32),
        "stage3_out": ("int", 32, 192, 32),
    }

@register_space("zhaocnn")
def _space_zhaocnn():
    return {
        "stage1_out": ("int", 32, 192, 32),
        "stage2_out": ("int", 32, 192, 32),
        "stage3_out": ("int", 32, 192, 32),
    }

@register_space("tcn")
def _space_tcn():
    return {
        "n_blocks":   ("int",      4,   8,  1),
        "n_filters":  ("int",     32, 192, 32),
        "kernel_size":("int",      3,   9,  2),
        "dropout":    ("float",  0.0, 0.4,0.1),
    }

@register_space("rnn_gru")
def _space_rnn_gru():
    return {
        "gru_hidden":   ("int",   16, 128, 16),
        "num_layers":   ("int",    1,   3,  1),
        "bidirectional":("choice",[True,False]),
        "dropout_rate": ("float",  0.0, 0.4,0.1),
    }

@register_space("rnn_lstm")
def _space_rnn_lstm():
    return {
        "lstm_hidden":  ("int",   16, 128, 16),
        "num_layers":   ("int",    1,   3,  1),
        "bidirectional":("choice",[True,False]),
        "dropout_rate": ("float",  0.0, 0.4,0.1),
    }

@register_space("peepholelstm")
def _space_peepholelstm():
    return {
        "hidden_size":  ("int",   16, 128, 16),
        "num_layers":   ("int",    1,   3,  1),
        "bidirectional":("choice",[True,False]),
    }

@register_space("phasedlstm")
def _space_phasedlstm():
    return {
        "hidden_size": ("int",   16, 128, 16),
        "num_layers":  ("int",    1,   3,  1),
        "period_init": ("float", 0.5, 5.0,0.5),
        "r_on_init":   ("float", 0.01,0.2,0.01),
    }

@register_space("indrnn")
def _space_indrnn():
    return {
        "hidden_size":  ("int",   16, 128,16),
        "num_layers":   ("int",    1,   3, 1),
        "dropout_rate": ("float",  0.0, 0.4,0.1),
        "clip_max":     ("float",  0.5, 2.0,0.5),
    }

@register_space("qrnn")
def _space_qrnn():
    return {
        "hidden_size":  ("int",   16, 128,16),
        "num_layers":   ("int",    1,   3, 1),
        "kernel_size":  ("int",    2,   8, 2),
        "dropout_rate": ("float",  0.0, 0.4,0.1),
        "pool_type":    ("choice",["fo","f","o"]),
    }

# ────────────────────────────────────────────────────────────────
# Espacios HPO restantes
# ────────────────────────────────────────────────────────────────

@register_space("esn")
def _space_esn():
    return {
        "reservoir_size":   ("int",       100, 1000, 100),
        "spectral_radius":  ("float",     0.1,   1.5,   0.1),
        "sparsity":         ("float",     0.0,   0.9,   0.1),
        "ridge_alpha":      ("loguniform", 1e-6,  1e-1),
    }

@register_space("tlgru")
def _space_tlgru():
    return {
        "hidden_size":   ("int",       32, 192, 32),
        "num_layers":    ("int",        1,   3,  1),
        "dropout":       ("float",     0.0, 0.4, 0.1),
    }

@register_space("vsgrunet")
def _space_vsgrunet():
    return {
        "hidden_size":   ("int",       32, 192, 32),
        "num_layers":    ("int",        1,   3,  1),
        "dropout":       ("float",     0.0, 0.4, 0.1),
    }

@register_space("deepresidualbilstm")
def _space_deepresidualbilstm():
    return {
        "hidden_dim":    ("int",       32, 192, 32),
        "num_layers":    ("int",        1,   5,  1),
        "dropout":       ("float",     0.0, 0.4, 0.1),
    }

@register_space("cnngru")
def _space_cnngru():
    return {
        "stage1_out":    ("int",       32, 192, 32),
        "stage2_out":    ("int",       32, 192, 32),
        "stage3_out":    ("int",       32, 192, 32),
        "stage4_out":    ("int",       32, 192, 32),
        "rnn_type":      ("choice",   ["GRU", "LSTM"]),
        "bidirectional": ("choice",   [True, False]),
        "dropout_rate":  ("float",     0.0, 0.4, 0.1),
        "se_ratio":      ("int",        8,  32,  8),
        "use_se":        ("choice",   [True, False]),
    }

@register_space("cnnlstm")
def _space_cnnlstm():
    return {
        "stage1_out":    ("int",       32, 192, 32),
        "stage2_out":    ("int",       32, 192, 32),
        "stage3_out":    ("int",       32, 192, 32),
        "stage4_out":    ("int",       32, 192, 32),
        "lstm_hidden":   ("int",       32, 128, 16),
        "bidirectional": ("choice",   [True, False]),
        "dropout_rate":  ("float",     0.0, 0.4, 0.1),
        "se_ratio":      ("int",        8,  32,  8),
        "use_se":        ("choice",   [True, False]),
    }

@register_space("cnnlstmgru")
def _space_cnnlstmgru():
    return {
        "stage1_out":         ("int",       32, 192, 32),
        "stage2_out":         ("int",       32, 192, 32),
        "stage3_out":         ("int",       32, 192, 32),
        "stage4_out":         ("int",       32, 192, 32),
        "lstm_hidden":        ("int",       32, 128, 16),
        "gru_hidden":         ("int",       32, 128, 16),
        "lstm_bidirectional": ("choice",   [True, False]),
        "gru_bidirectional":  ("choice",   [True, False]),
        "dropout_rate":       ("float",     0.0, 0.4, 0.1),
        "se_ratio":           ("int",        8,  32,  8),
        "use_se":             ("choice",   [True, False]),
    }

@register_space("disjointlstm")
def _space_disjointlstm():
    return {
        "residual":      ("choice",   [True, False]),
        "bidirectional": ("choice",   [True, False]),
        "hidden_size":   ("int",       32, 192, 32),
    }

@register_space("lstmfcn")
def _space_lstmfcn():
    return {
        "conv1_out":     ("int",       32, 192, 32),
        "conv2_out":     ("int",       32, 192, 32),
        "conv3_out":     ("int",       32, 192, 32),
        "lstm_hidden":   ("int",       32, 128, 16),
        "bidirectional": ("choice",   [True, False]),
        "dropout_rate":  ("float",     0.0, 0.4, 0.1),
    }

@register_space("cnnbilstmbigrudil")
def _space_cnnbilstmbigrudil():
    return {
        "cnn_channels":  ("int",       32, 192, 32),
        "gru_hidden":    ("int",       32, 128, 16),
        "lstm_hidden":   ("int",       32, 128, 16),
        "dropout":       ("float",     0.0, 0.4, 0.1),
    }
# ────────────────────────────────────────────────────────────────
# Attention based HPO spaces
# ────────────────────────────────────────────────────────────────

@register_space("alstm")
def _space_alstm():
    """Search-space para ALSTM."""
    return {
        "stage1_out":   ("int",       32, 192, 32),
        "stage2_out":   ("int",       32, 192, 32),
        "stage3_out":   ("int",       32, 192, 32),
        "lstm_hidden":  ("int",       32, 128, 16),
    }

@register_space("transformer1d")
def _space_transformer1d():
    """Search-space para Transformer1D."""
    return {
        "d_model":     ("int",      32, 256, 32),
        "num_heads":   ("choice", [2, 4, 8]),
        "num_layers":  ("choice", [1, 2, 3, 4, 5, 6]),
        "dropout":     ("float",    0.0, 0.4, 0.1),
    }

@register_space("informer1d")
def _space_informer1d():
    """Search-space para Informer1D."""
    return {
        "d_model":     ("int",      32, 256, 32),
        "num_heads":   ("choice", [2, 4, 8]),
        "num_layers":  ("choice", [1, 2, 3, 4, 5, 6]),
        "dropout":     ("float",    0.0, 0.4, 0.1),

    }

@register_space("autoformer1d")
def _space_autoformer1d():
    """Search-space para Autoformer1D."""
    return {
        "d_model":     ("int",      32, 256, 32),
        "num_heads":   ("choice", [2, 4, 8]),
        "num_layers":  ("choice", [1, 2, 3, 4, 6]),
        "dropout":     ("float",    0.0, 0.4, 0.1),
    }


@register_space("patchtst")
def _space_patchtst():
    return {
        # Valores seguros con T=256; evita 0 parches en casi todos los casos
        "patch_len":   ("choice", [8, 16, 32]),
        "stride":      ("choice", [4, 8, 16]),   # sin constraints dependientes
        "d_model":     ("int",    64, 256, 32),  # [64,96,128,160,192,224,256]
        "num_heads":   ("choice", [2, 4, 8]),
        "num_layers":  ("choice", [1, 2, 3, 4, 5, 6]),
        "dropout":     ("float",  0.0, 0.4, 0.1),
    }

@register_space("vit1d")
def _space_vit1d():
    """Search-space para ViT1D."""
    return {
        "d_model":     ("int",      32, 256, 32),
        "num_heads":   ("choice", [2, 4, 8]),
        "num_layers":  ("choice", [2, 4, 6]),
        "mlp_dim":     ("choice", [128, 256, 512]),
        "dropout":     ("float",    0.0, 0.4, 0.1),
        "patch_size":  ("choice", [8, 16, 32]),
    }

@register_space("timenet1d")
def _space_timenet1d():
    """Search-space para TimeNet1D."""
    return {
        "hidden_size": ("int",       32, 192, 32),
        "num_layers":  ("int",        1,   6,  1),
        "dropout":     ("float",     0.0, 0.4, 0.1),
    }

@register_space("physformer")
def _space_physformer():
    """
    Search-space para PhysMixFormer1D (atención multiescala + gauge + Π
    fila-estocástica + rama física de difusión Chebyshev, portado y
    validado desde aux/physformer_standalone/physmixformer_standalone_v3.py
    -- reemplaza al PhysFormer sin física que ocupaba esta variante antes).

    d_model/num_heads/num_layers/dropout: mismos rangos que el PhysFormer
    original (2026-08-05: ampliar d_model/num_layers empeoró el resultado
    en los 4 datasets con el mismo presupuesto de trials -- ver historial
    de esta función). pe_type/ms_type se dejan FIJOS en el código
    ("rope"/"inception", lo validado en el standalone) en vez de ser HPO
    -- no formaban parte de la matriz de ablación validada.

    ablation/fuse_physics NO son HPO (2026-08-11): con 15 trials y auckland
    teniendo solo 18 muestras totales (10/4/4 tras el split), BOHB no tiene
    señal suficiente para elegir de forma confiable si la física ayuda --
    en la práctica terminó escogiendo ablation=transformer/fuse_physics=False
    en auckland, es decir SIN física, contradiciendo la comparación contra
    el standalone (que siempre corre con physmixformer + física). Se fijan
    ambos en el código (PhysMixFormer1D.__init__ default: ablation=
    "physmixformer", fuse_physics=True) igual que el standalone, para que
    la física de verdad se use cuando la geometría está disponible. La
    física en sí (Π/L_e/gauge) se sigue auto-desactivando por dataset según
    disponibilidad de geometría de malla confiable -- ver
    data/geometry.py::load_geometry -- eso no cambia.

    n_cheb_bases/physics_warmup_epochs sí quedan como HPO: controlan la
    rama física (nº de bases de Chebyshev, épocas de warm-up del gate).
    """
    return {
        "d_model":               ("int",    64, 256, 32),
        "num_heads":              ("choice", [2, 4, 8, 16]),
        "num_layers":             ("choice", [1, 2, 3, 4]),
        "dropout":                ("float",  0.0, 0.4, 0.1),
        "n_cheb_bases":           ("choice", [3, 5, 7]),
        "physics_warmup_epochs":  ("choice", [1, 4, 8, 12]),
    }

@register_space("bertbigru")
def _space_bertbigru():
    """Search-space para BERT_BIGRU."""
    return {
        "hidden_dim":    ("int",       32, 192, 32),
        "num_layers":    ("choice",   [1, 2, 3]),
        "dropout_rate":  ("float",     0.0, 0.4, 0.1),
    }
