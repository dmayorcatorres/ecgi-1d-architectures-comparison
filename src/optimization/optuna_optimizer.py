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
        "disjointresnet" : lambda: {
            "n_feature_maps": trial.suggest_int("n_feature_maps", 32, 128, step=32),
        },
        "fcn": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32)
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
        "zhaocnn" : lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
        },

        # RNNs
        "cnngru": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnlstm" : lambda: { 
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnlstmgru": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 64, step=8),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnbilstm" : lambda: { 
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnbigru": lambda: { 
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
        },
        "cnnbilstmbigru": lambda: {
            "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
            "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
            "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
            "stage4_out": trial.suggest_int("stage4_out", 32, 256, step=32),
            "gru_hidden": trial.suggest_int("gru_hidden", 8, 64, step=8),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 64, step=8),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.8, step=0.1),
            "se_ratio": trial.suggest_int("se_ratio", 4, 32, step=4),
            "use_se": trial.suggest_categorical("use_se", [True, False]),
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