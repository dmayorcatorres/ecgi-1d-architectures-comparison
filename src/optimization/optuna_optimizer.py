import optuna
import logging
import torch
from ignite.engine import Events
from src.training.engines import TrainingEngine
from src.models.model_factory import ModelFactory
from typing import Dict, Any


logger = logging.getLogger(__name__)


class OptunaOptimizer:
    def __init__(self, config, data_loaders, model_factory):
        self.config = config
        self.data_loaders = data_loaders
        self.model_factory = model_factory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @staticmethod
    def sample_parameters(trial: optuna.Trial, variant: str) -> dict:
        # Implementar lógica original de HyperparameterOptimizer aquí
        param_map: Dict[str, Any] = {

        "disjointcnn": lambda: {
            "num_filters": trial.suggest_int("num_filters", 32, 128, step=32)
        },
        # -------------   DenseNet‑BiLSTM   ------------------------
        "densenetbilstm": lambda: {
            "initial_filters": trial.suggest_int("initial_filters", 16, 64, step=16),
            "growth_rate_small": trial.suggest_int("growth_rate_small", 16, 64, step=16),
            "growth_rate_large": trial.suggest_int("growth_rate_large", 32, 128, step=32),
            "num_layers_small":  trial.suggest_int("num_layers_small", 2, 4),
            "num_layers_large":  trial.suggest_int("num_layers_large", 1, 3),
            "transition_small":  trial.suggest_int("transition_small", 16, 64, step=16),
            "transition_large":  trial.suggest_int("transition_large", 32, 128, step=32),
        },

        "disjointresnet" : lambda: {
            "n_feature_maps": trial.suggest_int("n_feature_maps", 32, 128, step=32),
        },
        "fcn": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32)
        },
        "mlp": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 128, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 128, step=32)
        },
        "inceptiontime": lambda: {
            "num_filters": trial.suggest_int("num_filters", 32, 128, step=32),
            "bottleneck_channels": trial.suggest_int("bottleneck_channels", 16, 64, step=16)
        }, 
        "multichanneldcnn": lambda: {
            "sub_channels": trial.suggest_int("sub_channels", 32, 64, step=32)
        },
        "multiscalecnn" : lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
        },
        "multiscalecnn2": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),

            # **extras** específicos de esta versión
            "kernel_sizes": trial.suggest_categorical(
                "kernel_sizes",
                [(3, 5, 7), (2, 4, 8), (5, 7, 9)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        },

        "oscnn": lambda: {
            "base_channels": trial.suggest_int("base_channels", 2, 8, step=2),
        },


        "dilatedcnn": lambda: {
            "num_filters": trial.suggest_int("num_filters", 16, 32, step=8),
            "fc_units": trial.suggest_int("fc_units", 64, 128, step=32) 
        },
        "lite": lambda: {
            "n_filters": trial.suggest_int("n_filters", 32, 128, step=32),
            "base_kernel_size": trial.suggest_int("base_kernel_size", 20, 60, step=20),
            "use_custom_filters": trial.suggest_categorical("use_custom_filters", [True, False])
        },
        "resnet": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
        },
        "resnetcnn": lambda: {
            "conv_in_out": trial.suggest_int("conv_in_out", 32, 128, step=32),
        },
        "tfrcnn": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 128, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 128, step=32)
        },
        "zhaocnn" : lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
        },

        "rnn_gru": lambda trial: {
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 128, step=8),
            "num_layers": trial.suggest_int("gru_layers", 1, 3),
            "bidirectional": trial.suggest_categorical("gru_bidir", [False, True]),
            "dropout_rate": trial.suggest_float("gru_dropout", 0.0, 0.5, step=0.1),
        },
        "rnn_lstm": lambda trial: {
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 128, step=8),
            "num_layers": trial.suggest_int("lstm_layers", 1, 3),
            "bidirectional": trial.suggest_categorical("lstm_bidir", [False, True]),
            "dropout_rate": trial.suggest_float("lstm_dropout", 0.0, 0.5, step=0.1),
        
        },
        "peepholelstm": lambda trial: {
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "bidirectional": False,      # no bidireccional en este ejemplo
        },
        "phasedlstm": lambda trial: {
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
        },
        "indrnn": lambda trial: {
            # Número de unidades en cada capa IndRNN
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            # Número de capas
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            # Tasa de dropout entre capas recurrentes
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
            # Saturación máxima de los pesos recurrentes (para controlar explosión de gradiente)
            "clip_max": trial.suggest_float("clip_max", 0.5, 2.0, step=0.5),
        },
        "qrnn": lambda trial: {
            # Número de filtros en cada capa QRNN
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            # Número de capas QRNN
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            # Tamaño de la ventana convolucional (kernel) para la puerta forget
            "kernel_size": trial.suggest_int("kernel_size", 2, 4),
            # Tasa de dropout entre capas
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
            # Tipo de pooling para el z-dropout (si aplica)
            "pool_type": trial.suggest_categorical("pool_type", ["fo", "f", "o"]),
        },
        "esn": lambda trial: {
            # Tamaño del reservorio
            "reservoir_size": trial.suggest_int("reservoir_size", 100, 1000, step=100),
            # Radio espectral (spectral radius)
            "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5, step=0.1),
            # Grado de sparsity de la matriz de transición
            "sparsity": trial.suggest_float("sparsity", 0.0, 0.9, step=0.1),
            # Tasa de regularización de la regresión en la salida
            "ridge_alpha": trial.suggest_loguniform("ridge_alpha", 1e-6, 1e-1),
        },

        "tcn": lambda: {
            "n_blocks":  trial.suggest_int("n_blocks", 4, 8),
            "n_filters": trial.suggest_int("n_filters", 32, 128, step=32),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
        },
        "tlgru": lambda: {
            "hidden_size": trial.suggest_int("hidden_size", 32, 128, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        },
            # ─── VS-GRU (variable sensitive GRU) ─────────────────────────────
        "vsgrunet": lambda: {
            # tamaño de la representación oculta dentro de la GRU
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            # nº de capas GRU apiladas (1 = capa única ⇒ VS-GRU “puro”)
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            # regularización Drop-Out aplicada a la GRU
            "dropout":     trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        },

        # RNNs
        "cnngru": lambda trial: {
            "stage1_out":   trial.suggest_int("stage1_out",   32, 256, step=32),
            "stage2_out":   trial.suggest_int("stage2_out",   32, 256, step=32),
            "stage3_out":   trial.suggest_int("stage3_out",   32, 256, step=32),
            "stage4_out":   trial.suggest_int("stage4_out",   32, 256, step=32),
            "rnn_type":     trial.suggest_categorical("rnn_type", ["GRU", "LSTM"]),
            "rnn_hidden":   trial.suggest_int("rnn_hidden",    8,  64, step=8),
            "bidirectional":trial.suggest_categorical("bidirectional", [False, True]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio":     trial.suggest_int("se_ratio",       4,  32, step=4),
            "use_se":       trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnlstm": lambda trial: {
            "stage1_out":    trial.suggest_int("stage1_out",    32,256,32),
            "stage2_out":    trial.suggest_int("stage2_out",    32,256,32),
            "stage3_out":    trial.suggest_int("stage3_out",    32,256,32),
            "stage4_out":    trial.suggest_int("stage4_out",    32,256,32),
            "lstm_hidden":   trial.suggest_int("lstm_hidden",   8, 64, 8),
            "bidirectional": False,
            "dropout_rate":  trial.suggest_float("dropout_rate", 0.2,0.8,0.1),
            "se_ratio":      trial.suggest_int("se_ratio",       4, 32, 4),
            "use_se":        trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnlstmgru": lambda trial: {
            "stage1_out":          trial.suggest_int("stage1_out", 32,256,32),
            "stage2_out":          trial.suggest_int("stage2_out", 32,256,32),
            "stage3_out":          trial.suggest_int("stage3_out", 32,256,32),
            "stage4_out":          trial.suggest_int("stage4_out", 32,256,32),
            "lstm_hidden":         trial.suggest_int("lstm_hidden", 8, 64, 8),
            "gru_hidden":          trial.suggest_int("gru_hidden",  8, 64, 8),
            "lstm_bidirectional":  trial.suggest_categorical("lstm_bidirectional", [False, True]),
            "gru_bidirectional":   trial.suggest_categorical("gru_bidirectional",  [False, True]),
            "dropout_rate":        trial.suggest_float("dropout_rate", 0.2,0.8,0.1),
            "se_ratio":            trial.suggest_int("se_ratio",       4, 32, 4),
            "use_se":              trial.suggest_categorical("use_se",       [True, False]),
        },
        "cnnbilstmbigrudil": lambda: {
            "cnn_channels": trial.suggest_int("cnn_channels", 32, 256, step=32),
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 128, step=8),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 128, step=8),
            "dropout": trial.suggest_float("dropout", 0.1, 0.9, step=0.1),
        },
        "cnnalstm": lambda: { 
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 128, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 128, step=32),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 64, step=8) # <-- correcto
        },
        "disjointlstm" : lambda: {
            "num_filters": trial.suggest_int("num_filters", 32, 128, step=32),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 32, step=8)
        },
        "lstmfcn": lambda trial: {
            "conv1_out": trial.suggest_int("lstmfcn_conv1_out", 64, 256, step=64),
            "conv2_out": trial.suggest_int("lstmfcn_conv2_out", 64, 512, step=64),
            "conv3_out": trial.suggest_int("lstmfcn_conv3_out", 64, 256, step=64),
            "lstm_hidden": trial.suggest_int("lstmfcn_lstm_hidden", 32, 256, step=32),
            "bidirectional": trial.suggest_categorical("lstmfcn_bidirectional", [False, True]),
            "dropout_rate": trial.suggest_float("lstmfcn_dropout_rate", 0.1, 0.8, step=0.1),
            "lr": trial.suggest_loguniform("lstmfcn_lr", 1e-4, 1e-2),
        },
        # Por esto (nota que ahora es "bertbigru" y la lambda no incluye el parámetro trial):
        "bertbigru": lambda: {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3]),
            "dropout_prob": trial.suggest_float("dropout_prob", 0.1, 0.5, step=0.1),
        },
        "deepresidualbilstm": lambda: {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=32),
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5)
        }
        } 

        if variant not in param_map:
            raise ValueError(f"Variante desconocida: {variant}")
        
        params = param_map[variant]()
        logger.info(f"Hiperparámetros muestreados para {variant}: {params}")
        return params
    
    def _objective(self, trial: optuna.Trial, variant: str) -> float:
        params = self.sample_parameters(trial, variant)
        model = self.model_factory.create_model(variant, params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float(f"lr_{variant}", 1e-4, 1e-2, log=True))
        
        trainer = TrainingEngine(model, optimizer, self.device)
        train_engine, val_engine = trainer.create_engines()
        
        @train_engine.on(Events.EPOCH_COMPLETED)
        def validate_model(engine):
            val_engine.run(self.data_loaders[1])
        
        train_engine.run(self.data_loaders[0], max_epochs=self.config.optuna_epochs)
        return val_engine.state.metrics['loss']

    def optimize(self, variants: list) -> dict:
        best_params = {}
        for variant in variants:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: self._objective(trial, variant), 
                n_trials=self.config.optuna_trials
            )
            best_params[variant] = study.best_params
        return best_params