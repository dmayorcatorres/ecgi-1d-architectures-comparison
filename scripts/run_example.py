import sys
import os
import math
import logging
import datetime
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import torch
from torchviz import make_dot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import logging
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import optuna
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import Loss

# Habilitar cudnn.benchmark si los tamaños de entrada son fijos
torch.backends.cudnn.benchmark = True

# --- Ajuste de PYTHONPATH para entornos locales ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ecgi.ECGiDataset import ECGiDataset, DatasetConfig
    from src.ecgi.ECGMetricsComparator import ECGMetricsComparator
    from src.ecgi.ECGiVisualizer import ECGiVisualizer
    from src.ecgi.SignalProcessorTorch import SignalProcessorTorch
    from src.models.cnn_models import RawCNNBase
    from src.ecgi.ResultsSaver import ResultsSaver
    from src.ecgi.TimeSeriesTo2DTransform import TimeSeriesTo2DTransform
except ModuleNotFoundError as e:
    logging.error("No se pudo encontrar el paquete 'src'. Verifica la estructura del proyecto y el PYTHONPATH.")
    raise e

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

###############################################################################
# Configuración del entrenamiento
###############################################################################
@dataclass
class TrainingConfig:
    data_path: str = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kitpvc", "kitpvc_data.mat")
    batch_size: int = 1
    optuna_epochs: int = 1      # Épocas por trial en la optimización
    final_epochs: int = 1       # Épocas en el entrenamiento final
    optuna_trials: int = 1       # Número de trials a ejecutar
    # Lista de variantes a probar
    #variants: Tuple[str, ...] = (  "res-cnn", "lstm-fcn", "lstm-disjoint", "disjoint-resnet", "lite",)
    variants: Tuple[str, ...] = ("fcn", "resnet", "dilated", "disjoint", "mc-dcnn", "mc-cnn", "inception", "lstm-fcn", "lstm-disjoint", "disjoint-resnet", "lite")
    #variants: Tuple[str, ...] = ("res-cnn", "mccnn","zhmc-cnn",)
    best_variant: str = "lite"
    verbose: bool = False         # Controla la verbosidad de los logs

###############################################################################
# Clases auxiliares: DeviceManager, HyperparameterOptimizer, ModelTrainer
###############################################################################
class DeviceManager:
    @staticmethod
    def get_device() -> torch.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Dispositivo usado: {device}")
        return device


logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Optimizer de hiperparámetros para modelos CNN en series temporales.
    
    Esta clase permite muestrear y parsear hiperparámetros específicos para cada variante de
    modelo utilizando Optuna. Las variantes disponibles son, por ejemplo:
      - "fcn"
      - "resnet"
      - "dilated"
      - "disjoint"
    
    Las funciones de muestreo y parseo se adaptan para que los hiperparámetros se pasen de
    manera elegante al modelo.
    """
    
    @staticmethod
    def sample_hyperparams(trial: optuna.Trial, variant: str) -> Dict[str, Any]:
        """
        Muestrea hiperparámetros específicos para la variante del modelo utilizando Optuna.
        
        Parámetros:
          - trial: Objeto de ensayo de Optuna.
          - variant (str): Variante del modelo (ej., "fcn", "resnet", "dilated", "disjoint").
        
        Retorna:
          - Dict[str, Any]: Diccionario de hiperparámetros sugeridos para la variante.
        """
        variant = variant.lower()
        param_map: Dict[str, Any] = {
            "fcn": lambda: {
                "stage1_out": trial.suggest_int("stage1_out", 32, 128, step=32)
            },
            "resnet": lambda: {
                "stage1_out": trial.suggest_int("stage1_out_resnet", 32, 128, step=32),
                "stage2_out": trial.suggest_int("stage2_out_resnet", 64, 256, step=32),
                "stage3_out": trial.suggest_int("stage3_out_resnet", 64, 256, step=32)
            },
            "dilated": lambda: {
                "num_filters": trial.suggest_int("num_filters", 32, 64, step=32),
                "fc_units": trial.suggest_int("fc_units", 64, 512, step=64) 
            },
            "disjoint": lambda: {
                "num_filters": trial.suggest_int("num_filters", 32, 128, step=32)
            },
            "mc-dcnn": lambda: {
                "sub_channels": trial.suggest_int("sub_channels", 32, 64, step=32)
            },
            "res-cnn": lambda: {
                "conv_in_out": trial.suggest_int("conv_in_out", 32, 128, step=32),
            },
            "mc-cnn" : lambda: {
                "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
                "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
                "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
            },
            "zhmc-cnn" : lambda: {
                "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
                "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
                "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32)
            },
            "inception": lambda: {
                "num_filters": trial.suggest_int("num_filters", 32, 128, step=32),
                "bottleneck_channels": trial.suggest_int("bottleneck_channels", 16, 64, step=16)
            },
            "lstm-fcn" : lambda: {                
                "stage1_out": trial.suggest_int("stage1_out", 32, 256, step=32),
                "stage2_out": trial.suggest_int("stage2_out", 32, 256, step=32),
                "stage3_out": trial.suggest_int("stage3_out", 32, 256, step=32),
                "lstm_out": trial.suggest_int("lstm_out", 8, 64, step=8)
            },
            "lstm-disjoint" : lambda: {
                "num_filters": trial.suggest_int("num_filters", 32, 128, step=32),
                "lstm_hidden": trial.suggest_int("lstm_hidden", 8, 32, step=8)
            },
            "disjoint-resnet" : lambda: {
                "n_feature_maps": trial.suggest_int("n_feature_maps", 32, 128, step=32),
            },
            "lite": lambda: {
                "n_filters": trial.suggest_int("n_filters", 32, 128, step=32),
                "base_kernel_size": trial.suggest_int("base_kernel_size", 20, 60, step=20),
                "use_custom_filters": trial.suggest_categorical("use_custom_filters", [True, False])
            }

        }
        
        if variant not in param_map:
            raise ValueError(f"Variante desconocida: {variant}")
        params = param_map[variant]()
        logger.info(f"Hiperparámetros muestreados para {variant}: {params}")
        return params

    @staticmethod
    def parse_params(params: Dict[str, Any], variant: str) -> Tuple[float, Dict[str, Any]]:
        """
        Extrae la tasa de aprendizaje y limpia las claves de los hiperparámetros específicos de la variante.

        Se busca la tasa de aprendizaje bajo la clave 'lr_{variant}'. Las demás claves que terminen
        en '_{variant}' se reformatean eliminando dicho sufijo.
        
        Parámetros:
          - params: Diccionario con los hiperparámetros muestreados.
          - variant (str): Variante del modelo (ej., "fcn", "resnet", etc.)
        
        Retorna:
          - Tuple[float, Dict[str, Any]]: La tasa de aprendizaje y un diccionario con los hiperparámetros limpiados.
        """
        variant = variant.lower()
        lr_key = f"lr_{variant}"
        lr = params.get(lr_key, 1e-3)
        overrides = {}
        suffix = f"_{variant}"
        for k, v in params.items():
            if k.startswith(lr_key):
                continue
            # Si la clave termina con el sufijo, se elimina; de lo contrario, se conserva tal cual.
            if k.endswith(suffix):
                new_key = k[:-len(suffix)]
                overrides[new_key] = v
            else:
                overrides[k] = v
        logger.info(f"Parámetros parseados para {variant}: lr={lr}, overrides={overrides}")
        return lr, overrides

class ModelTrainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.criterion = nn.MSELoss()

    def create_engines(self, model: nn.Module, optimizer: optim.Optimizer) -> Tuple[Engine, Engine]:
        def train_step(engine, batch):
            model.train()
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = self.criterion(preds, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                return model(inputs), targets

        train_engine = Engine(train_step)
        val_engine = Engine(eval_step)
        Loss(self.criterion).attach(val_engine, 'loss')
        return train_engine, val_engine

###############################################################################
# ExperimentRunner: Optimización, entrenamiento final y evaluación
###############################################################################
class ExperimentRunner:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = DeviceManager.get_device()
        self.ecgi_dataset = ECGiDataset(config.data_path, device=self.device)
        self._prepare_data()

    def _prepare_data(self):
        logger.info("Iniciando preparación de datos...")
        self.emg_tensor = self.ecgi_dataset.emg_tensor.cpu()
        self.ecg_tensor = self.ecgi_dataset.ecg_tensor.cpu()
        
        # División secuencial: 80% entrenamiento, 20% validación
        split_idx = int(0.8 * len(self.emg_tensor))
        train_emg, val_emg = self.emg_tensor[:split_idx], self.emg_tensor[split_idx:]
        train_ecg, val_ecg = self.ecg_tensor[:split_idx], self.ecg_tensor[split_idx:]
        # imprimir dimensiones
        print(train_emg.shape)
        print(val_emg.shape)
        print(train_ecg.shape)
        print(val_ecg.shape)

        self.train_dataset = TensorDataset(train_emg, train_ecg)
        self.val_dataset = TensorDataset(val_emg, val_ecg)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=True)

        self.input_channels = self.emg_tensor.shape[1]
        self.output_channels = self.ecg_tensor.shape[1]
        self.time_steps = self.emg_tensor.shape[2]

        
        logger.info("Preparación de datos completada.")

    def _run_trial(self, trial: optuna.Trial, variant: str) -> float:
        logger.info(f"Iniciando trial para variante {variant}")
        lr = trial.suggest_float(f"lr_{variant}", 1e-4, 1e-2, log=True)
        hp_overrides = HyperparameterOptimizer.sample_hyperparams(trial, variant)
        model = RawCNNBase(variant=variant,
                           time_steps=self.time_steps,
                           input_channels=self.input_channels,
                           output_channels=self.output_channels,
                           **hp_overrides).to(self.device)
        trainer = ModelTrainer(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_engine, _ = trainer.create_engines(model, optimizer)
        train_loss = []

        @train_engine.on(Events.EPOCH_COMPLETED)
        def log_epoch(engine):
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    preds = model(inputs)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
            val_mse = F.mse_loss(torch.cat(all_preds), torch.cat(all_targets)).item()
            # Se almacena como tupla para el trial (para Optuna) pero en entrenamiento final se usan listas de floats.
            train_loss.append((engine.state.epoch, engine.state.output, val_mse))
            logger.info(f"Epoch {engine.state.epoch} | Train Loss: {engine.state.output:.4f} | Val Loss: {val_mse:.4f}")

        train_engine.run(self.train_loader, max_epochs=self.config.optuna_epochs)
        logger.info("Trial finalizado.")
        return train_loss[-1][2]

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        logger.info("Iniciando la optimización de hiperparámetros...")
        best_results = {}
        for var in self.config.variants:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=2))
            study.optimize(lambda trial: self._run_trial(trial, var),
                           n_trials=self.config.optuna_trials, timeout=3600)
            best_results[var] = {'best_value': study.best_value, 'params': study.best_params}
            logger.info(f"Mejor resultado para {var}: {best_results[var]}")
        return best_results

    def _compute_metrics(self, targets: torch.Tensor, preds: torch.Tensor) -> Dict[str, Any]:
        comparator = ECGMetricsComparator(fs=1000)
        comparator.set_time_domain_signals(targets, preds)
        metrics = {
            'MAE_mean': comparator.compute_time_mae()[0],
            'MAE_std': comparator.compute_time_mae()[1],
            'CC_mean': comparator.compute_time_cc()[0],
            'CC_std': comparator.compute_time_cc()[1],
            'MSE_mean': comparator.compute_time_mse_rmse()[0][0],
            'MSE_std': comparator.compute_time_mse_rmse()[0][1],
            'RMSE_mean': comparator.compute_time_mse_rmse()[1][0],
            'RMSE_std': comparator.compute_time_mse_rmse()[1][1],
            'SNR_mean': comparator.compute_time_snr()[0],
            'SNR_std': comparator.compute_time_snr()[1]
        }
        logger.info(f"Métricas calculadas: {metrics}")
        return metrics

    def final_training(self, best_results: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Iniciando el entrenamiento final...")
        final_results = {}
        for var in self.config.variants:
            logger.info(f"Entrenamiento final para variante {var}")
            lr, hp_overrides = HyperparameterOptimizer.parse_params(best_results[var]['params'], var)
            model = RawCNNBase(variant=var,
                                time_steps=self.time_steps,
                                input_channels=self.input_channels,
                                output_channels=self.output_channels,
                                **hp_overrides).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            trainer = ModelTrainer(self.device)
            train_engine, val_engine = trainer.create_engines(model, optimizer)
            train_timer = Timer(average=True)
            train_timer.attach(train_engine, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)
            val_timer = Timer(average=True)
            val_timer.attach(val_engine, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)
            # Aquí se almacenan listas de floats para las curvas de aprendizaje.
            train_loss_history, val_loss_history = [], []

            @train_engine.on(Events.EPOCH_COMPLETED)
            def log_epoch(engine):
                val_engine.run(self.val_loader)
                val_metrics = val_engine.state.metrics
                train_loss = engine.state.output
                val_loss = val_metrics['loss']
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                logger.info(f"Epoch {engine.state.epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            train_engine.run(self.train_loader, max_epochs=self.config.final_epochs)
            if not val_loss_history:
                logger.error(f"No se registraron métricas de validación para {var}")
                continue

            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    preds = model(inputs)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
            final_metrics = self._compute_metrics(torch.cat(all_targets), torch.cat(all_preds))

                        # --- NUEVAS MÉTRICAS DE COMPLEJIDAD COMPUTACIONAL ---
            # Calcular el número de parámetros entrenables
              # --- NUEVAS MÉTRICAS DE COMPLEJIDAD COMPUTACIONAL ---
            # Calcular el número de parámetros entrenables:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Intentar calcular FLOPs y MACs usando la librería thop
            try:
                from thop import profile
                dummy_input = torch.randn(self.config.batch_size, self.input_channels, self.time_steps).to(self.device)
                flops, macs = profile(model, inputs=(dummy_input,), verbose=False)
                flops_str = f"{flops} FLOPs"
                macs_str = f"{macs} MACs"
            except ImportError:
                logger.warning("La librería thop no está instalada. Instale thop (pip install thop) para obtener FLOPs/MACs.")
                flops_str, macs_str = "N/A", "N/A"

            # Medir latencia de inferencia (en milisegundos)
            import time
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

                        
            final_results[var] = {
                'model': model,
                'Val Loss': val_loss_history[-1],
                **final_metrics,
                'Best Params': best_results[var]['params'],
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'Train Time (s)': train_timer.value() * self.config.final_epochs,
                'Val Time (s)': val_timer.value() * self.config.final_epochs,
                'num_params': num_params,          # previamente calculado
                'FLOPs': flops_str,                # previamente calculado o "N/A"
                'MACs': macs_str,                  # previamente calculado o "N/A"
                'inference_latency_ms': latency_ms, # previamente calculado
            }


            logger.info(f"Entrenamiento final completado para {var}")
        return final_results

    def load_all_models(self) -> Dict[str, nn.Module]:
        models = {}
        for var in self.config.variants:
            model = RawCNNBase(variant=var,
                                time_steps=self.time_steps,
                                input_channels=self.input_channels,
                                output_channels=self.output_channels).to(self.device)
            models[var] = model
        return models

###############################################################################
# ExperimentManager: Flujo completo del experimento y guardado organizado
###############################################################################
class ExperimentManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.runner = ExperimentRunner(config)
        # Se utiliza la carpeta "experiment_results/cnns" como base de resultados.
        self.saver = ResultsSaver(base_dir=os.path.join(os.path.dirname(__file__), "..", "experiment_results", "cnns"))
        # Lista para acumular las métricas de todas las variantes.
        self.all_metrics = []

    def run_experiment(self):
        logger.info("=== Iniciando el experimento completo ===")

        # 1) Optimización de hiperparámetros
        best_results = self.runner.optimize_hyperparameters()
        
        # 2) Entrenamiento final
        final_results = self.runner.final_training(best_results)
        
        # 3) Para cada variante, guardar resultados y recolectar métricas
        for variant, variant_results in final_results.items():
            self.save_variant_outputs(variant, variant_results)
            # Se acumulan las métricas para el CSV resumen
            metrics_data = {
                'Variant': variant,
                'Val Loss': variant_results.get('Val Loss', None),
                'Train Time (s)': variant_results.get('Train Time (s)', None),
                'Val Time (s)': variant_results.get('Val Time (s)', None),
                'Best Params': variant_results.get('Best Params', None),
                'MAE_mean': variant_results.get('MAE_mean', None),
                'MAE_std': variant_results.get('MAE_std', None),
                'CC_mean': variant_results.get('CC_mean', None),
                'CC_std': variant_results.get('CC_std', None),
                'MSE_mean': variant_results.get('MSE_mean', None),
                'MSE_std': variant_results.get('MSE_std', None),
                'RMSE_mean': variant_results.get('RMSE_mean', None),
                'RMSE_std': variant_results.get('RMSE_std', None),
                'SNR_mean': variant_results.get('SNR_mean', None),
                'SNR_std': variant_results.get('SNR_std', None),
                'num_params': variant_results.get('num_params', None),
                'FLOPs': variant_results.get('FLOPs', None),
                'MACs': variant_results.get('MACs', None),
                'gpu_memory_allocated': variant_results.get('gpu_memory_allocated', None),
                'gpu_memory_reserved': variant_results.get('gpu_memory_reserved', None),
                'inference_latency_ms': variant_results.get('inference_latency_ms', None),
                'model_size_mb': variant_results.get('model_size_mb', None)  # Usar el valor almacenado en variant_results
            }
            self.all_metrics.append(metrics_data)

        # 4) Crear DataFrame con todas las métricas de esta corrida
        df_all = pd.DataFrame(self.all_metrics)

        # 5) Agregar columna con el nombre de la carpeta actual (timestamp, etc.)
        folder_name = os.path.basename(self.saver.run_dir)  
        df_all['FolderName'] = folder_name

        # 6) Guardar CSV local (subcarpeta del experimento)
        self.saver.save_csv(df_all, filename="all_metrics.csv")

        # 7) Actualizar o crear Excel acumulativo en la carpeta base "cnns"
        accumulative_xlsx_path = os.path.join(
            os.path.dirname(self.saver.run_dir),  # la carpeta "cnns"
            "accumulative_metrics.xlsx"
        )

        if os.path.isfile(accumulative_xlsx_path):
            old_df = pd.read_excel(accumulative_xlsx_path)
            new_df = pd.concat([old_df, df_all], ignore_index=True)
        else:
            new_df = df_all

        new_df.to_excel(accumulative_xlsx_path, index=False)
        logger.info(f"Acumulado actualizado en Excel: {accumulative_xlsx_path}")

        # 8) Guardar gráficos comunes de transformaciones en la carpeta principal
        self.save_common_plots()

        logger.info("=== Experimento completado ===")

    def save_variant_outputs(self, variant: str, variant_results: Dict[str, Any]):
        """
        Crea una carpeta para la variante y guarda:
          - Curva de aprendizaje
          - Modelo (.pt)
          - Exporta el modelo en formato ONNX y TorchScript
          - Gráfico de la arquitectura (con HiddenLayer)
          - Gráfico de electrodos (señales reales vs. predichas)
        """
        variant_dir = os.path.join(self.saver.run_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)
        logger.info(f"Guardando resultados para la variante {variant} en {variant_dir}")

        # Curva de aprendizaje
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(variant_results['train_loss_history'], label='Train Loss')
        ax.plot(variant_results['val_loss_history'], label='Val Loss')
        ax.set_title(f"Learning Curve - {variant}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        fig.tight_layout()
        learning_curve_file = os.path.join(variant_dir, "learning_curve.png")
        fig.savefig(learning_curve_file)
        plt.close(fig)

        # Guardar modelo (.pt)
        model_file = os.path.join(variant_dir, f"{variant}_model.pt")
        torch.save(variant_results['model'].state_dict(), model_file)

        # Calcular el tamaño del modelo en disco y almacenarlo en variant_results
        try:
            model_size_bytes = os.path.getsize(model_file)
            model_size_mb = model_size_bytes / (1024 * 1024)
            variant_results['model_size_mb'] = model_size_mb
            logger.info(f"Tamaño del modelo en disco: {model_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"Error calculando el tamaño del modelo: {str(e)}")
            variant_results['model_size_mb'] = None

        # Exportar a ONNX
        dummy_input = torch.randn(1, self.runner.input_channels, self.runner.time_steps).to(self.runner.device)
        model = variant_results['model'].eval()
        onnx_file = os.path.join(variant_dir, f"{variant}_model.onnx")
        try:
            torch.onnx.export(
                model,
                dummy_input, 
                onnx_file,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"]
            )
            logger.info(f"Modelo guardado en formato ONNX: {onnx_file}")
        except Exception as e:
            logger.error(f"Error exportando ONNX para {variant}: {str(e)}")

        # Exportar a TorchScript (tracing)  
        ts_file = os.path.join(variant_dir, f"{variant}_model_torchscript.pt")
        try:
            traced_script = torch.jit.trace(model, dummy_input)
            traced_script.save(ts_file)
            logger.info(f"Modelo guardado en formato TorchScript: {ts_file}") 
        except Exception as e:
            logger.error(f"Error exportando TorchScript para {variant}: {str(e)}")

        # Gráfico de electrodos: señales reales vs. predichas
        full_dataset = torch.utils.data.TensorDataset(self.runner.emg_tensor, self.runner.ecg_tensor)
        full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=self.config.batch_size, shuffle=False)
        variant_results['model'].eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in full_loader:
                inputs = inputs.to(self.runner.device)
                preds = variant_results['model'](inputs)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        pred_tensor = torch.cat(all_preds)
        real_tensor = torch.cat(all_targets)
        electrode_file = os.path.join(variant_dir, "electrode_plot.png")
        ECGiVisualizer.plot_electrodes(
            original_data_3d=real_tensor,
            reconstructed_data_3d=pred_tensor,
            fs=1000,
            signal_label="ECG",
            num_to_plot=real_tensor.shape[0],
            save_path=electrode_file
        )
            
        logger.info(f"Resultados para {variant} guardados en {variant_dir}")


    def save_common_plots(self):
        """
        Guarda ejemplos de gráficos de múltiples transformaciones 
        (RPM, GAF, RP, JRP, CWT, DWT, STFT, Spectrogram, MTF)
        en la carpeta principal de resultados.
        """
        transformer = TimeSeriesTo2DTransform(fs=1000.0)
        visualizer = ECGiVisualizer(fs=1000)
        
        # Tomamos data_3d (ecg_tensor) como ejemplo
        data_3d = self.runner.ecgi_dataset.ecg_tensor

        # Ejemplo: Escogemos la señal 0 (signal_index=0), y submuestreamos 6 electrodos
        sig_idx = 0
        num_electrodes_to_plot = 6

        # 1) RPM
        rpm_tensor = transformer.transform(x=data_3d, method='rpm')
        rpm_file = os.path.join(self.saver.run_dir, "plot_rpm.png")
        visualizer.visualize_rpm(
            rpm_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='RPM Example',
            save_path=rpm_file
        )

        # 2) GAF (ejemplo con summation)
        gaf_tensor = transformer.transform(x=data_3d, method='gaf', gaf_mode='summation')
        gaf_file = os.path.join(self.saver.run_dir, "plot_gaf_summation.png")
        visualizer.visualize_gaf(
            gaf_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='GAF Summation',
            save_path=gaf_file
        )

        # 3) MTF
        mtf_tensor = transformer.transform(x=data_3d, method='mtf', n_bins=8)
        mtf_file = os.path.join(self.saver.run_dir, "plot_mtf.png")
        visualizer.visualize_mtf(
            mtf_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='MTF Example',
            save_path=mtf_file
        )

        # 4) RP
        rp_tensor = transformer.transform(x=data_3d, method='rp', threshold=0.2)
        rp_file = os.path.join(self.saver.run_dir, "plot_rp.png")
        visualizer.visualize_rp(
            rp_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='RP Example',
            save_path=rp_file
        )

        # 5) JRP
        jrp_tensor = transformer.transform(x=data_3d, method='jrp', embedding_dim=3, time_delay=1, threshold=0.1)
        jrp_file = os.path.join(self.saver.run_dir, "plot_jrp.png")
        visualizer.visualize_jrp(
            jrp_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='JRP Example',
            save_path=jrp_file
        )

        # 6) STFT
        stft_tensor = transformer.transform(x=data_3d, method='stft', window_type='hann', nperseg=256)
        stft_file = os.path.join(self.saver.run_dir, "plot_stft.png")
        visualizer.visualize_stft(
            stft_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='STFT Example',
            save_path=stft_file
        )

        # 7) Spectrogram
        spectro_tensor = transformer.transform(x=data_3d, method='spectrogram', nperseg=32, noverlap=16)
        spectro_file = os.path.join(self.saver.run_dir, "plot_spectrogram.png")
        visualizer.visualize_spectrogram(
            spectro_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='Spectrogram Example',
            save_path=spectro_file
        )

        # 8) CWT
        cwt_tensor = transformer.transform(x=data_3d, method='cwt', wavelet_type='morlet')
        cwt_file = os.path.join(self.saver.run_dir, "plot_cwt.png")
        visualizer.visualize_cwt(
            cwt_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            title_label='CWT Example',
            save_path=cwt_file
        )

        # 9) DWT
        dwt_tensor = transformer.transform(x=data_3d, method='dwt', wavelet='db4')
        dwt_file = os.path.join(self.saver.run_dir, "plot_dwt.png")
        ECGiVisualizer.visualize_dwt(
            dwt_tensor,
            signal_index=sig_idx,
            num_to_plot=num_electrodes_to_plot,
            wavelet_label='DWT Example',
            save_path=dwt_file
        )

        logger.info(f"Se han guardado los gráficos de RPM, GAF, RP, JRP, CWT, DWT, STFT, Spectrogram y MTF en {self.saver.run_dir}")

###############################################################################
# MAIN
###############################################################################
def main():
    config = TrainingConfig()
    if config.verbose:
        logging.getLogger().setLevel(logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.WARNING)
    exp_manager = ExperimentManager(config)
    exp_manager.run_experiment()

if __name__ == "__main__":
    main()