# run_trials.py (archivo principal)
from __future__ import annotations
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset, Subset
from ignite.engine import Events
import ray

# Configuración de paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Importaciones locales
from src.configs.base_config import TrainingConfig
from src.data.datasets import ECGDataLoaderFactory
from src.models.model_factory import ModelFactory
from src.training.engines import TrainingEngine
from src.utils.visualization import ResultsVisualizer  # <-- Importar desde su módulo
from src.ecgi.ECGMetricsComparator import ECGMetricsComparator

# Configuración de logging
logger = logging.getLogger(__name__)


class DataHandler:
    """Maneja la carga y validación de datos"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.factory = ECGDataLoaderFactory(config)
        
    def load_data(self) -> Dict[str, DataLoader]:
        """Carga y valida los datos"""
        logger.info("Inicializando cargadores de datos...")
        train_loader, val_loader = self.factory.create_loaders()
        self._update_config_with_sample_data(train_loader)
        self._validate_data_shapes(train_loader, 'train')
        self._validate_data_shapes(val_loader, 'val')
        return {'train': train_loader, 'val': val_loader}
    
    def _update_config_with_sample_data(self, loader: DataLoader):
        sample_ecg, sample_emg = next(iter(loader))
        self.config.input_channels = sample_ecg.size(1)
        self.config.output_channels = sample_emg.size(1)
        self.config.time_steps = sample_ecg.size(2)
        logger.debug(f"Config actualizada: {self.config}")

    def _validate_data_shapes(self, loader: DataLoader, name: str, n_batches: int = 1):
        """Valida las dimensiones de los datos"""
        for idx, (inp, tgt) in enumerate(loader):
            self._check_shapes(inp, tgt, name, idx)
            if idx+1 >= n_batches:
                break
        logger.info(f"[{name}] Validación de shapes exitosa")

    def _check_shapes(self, inp: torch.Tensor, tgt: torch.Tensor, name: str, idx: int):
        errors = []
        if inp.size(-1) != tgt.size(-1):
            errors.append(f"Mismatch temporal: {inp.size(-1)} vs {tgt.size(-1)}")
        if inp.size(1) != self.config.input_channels:
            errors.append(f"Canales input: {inp.size(1)} vs {self.config.input_channels}")
        if tgt.size(1) != self.config.output_channels:
            errors.append(f"Canales target: {tgt.size(1)} vs {self.config.output_channels}")
        
        if errors:
            raise ValueError(f"[{name}] Batch {idx}: {'; '.join(errors)}")


class ExperimentSetup:
    """Configura el entorno del experimento"""
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.current_run_dir: Optional[Path] = None
        
    def create_run_directory(self, experiment_name: str) -> Path:
        """Crea directorio único para el experimento"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        return self.current_run_dir


class ModelEvaluator:
    """Evalúa modelos y calcula métricas"""
    def __init__(self, device: torch.device):
        self.device = device
        
    def compute_model_stats(self, model: torch.nn.Module, config: TrainingConfig) -> Dict[str, Any]:
        """Calcula estadísticas del modelo"""
        stats = {
            'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'inference_latency_ms': self._measure_latency(model, config),
            **self._calculate_flops(model, config)
        }
        logger.debug(f"Estadísticas del modelo: {stats}")
        return stats
    
    def _measure_latency(self, model: torch.nn.Module, config: TrainingConfig) -> float:
        dummy = torch.randn(1, config.input_channels, config.time_steps).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        return (time.perf_counter() - start) * 1000
    
    def _calculate_flops(self, model: torch.nn.Module, config: TrainingConfig) -> Dict[str, Any]:
        try:
            from thop import profile
            dummy = torch.randn(1, config.input_channels, config.time_steps).to(self.device)
            flops, macs = profile(model, inputs=(dummy,), verbose=False)
            return {'FLOPs': flops, 'MACs': macs}
        except ImportError:
            logger.warning("thop no instalado, FLOPs/MACs no disponibles")
            return {'FLOPs': None, 'MACs': None}

    def collect_predictions(
        self, 
        model: torch.nn.Module, 
        loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recolecta predicciones y targets, asegurando forma (B, C, T).
        """
        model.eval()
        all_preds, all_tgts = [], []

        with torch.no_grad():
            for batch_idx, (inp, tgt) in enumerate(loader):
                inp = inp.to(self.device, non_blocking=True)
                pred = model(inp).cpu()  # (B, C_out, T) esperado

                # Si tgt está en (B, T, C_out), permutamos
                if tgt.ndim == 3 and tgt.shape[1] != pred.shape[1]:
                    tgt = tgt.permute(0, 2, 1)

                all_preds.append(pred)
                all_tgts.append(tgt)

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        return torch.cat(all_preds), torch.cat(all_tgts)

    def evaluate_model(
        self, 
        model: torch.nn.Module, 
        loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evalúa el modelo y devuelve un dict con las 8 métricas:
         MAE_mean, MAE_std, RMSE_mean, RMSE_std,
         MSE_mean, MSE_std, CC_mean, CC_std.
        En caso de error, todos los valores serán NaN.
        """
        try:
            preds, targets = self.collect_predictions(model, loader)
            return self._compute_metrics(targets, preds)

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            nan = float('nan')
            return {
                "MAE_mean":  nan, "MAE_std":  nan,
                "RMSE_mean": nan, "RMSE_std": nan,
                "MSE_mean":  nan, "MSE_std":  nan,
                "CC_mean":   nan, "CC_std":   nan,
            }


    def _compute_metrics(
        self, 
        targets: torch.Tensor, 
        preds: torch.Tensor
    ) -> Dict[str, float]:
        """
        Llama a ECGMetricsComparator y extrae las 8 métricas.
        """
        cmp = ECGMetricsComparator(fs=1000)
        cmp.set_time_domain_signals(targets, preds)

        mae_mean,  mae_std        = cmp.compute_time_mae()
        cc_mean,   cc_std         = cmp.compute_time_cc()
        (mse_mean, mse_std),      \
        (rmse_mean, rmse_std)     = cmp.compute_time_mse_rmse()

        return {
            "MAE_mean":  mae_mean,    "MAE_std":  mae_std,
            "CC_mean":   cc_mean,     "CC_std":   cc_std,
            "MSE_mean":  mse_mean,    "MSE_std":  mse_std,
            "RMSE_mean": rmse_mean,   "RMSE_std": rmse_std,
        }

    
class CheckpointManager:
    """Maneja la carga/guardado de checkpoints"""
    def __init__(self, path: Path):
        self.path = path
        
    def load_checkpoint(self, visualizer: ResultsVisualizer) -> Optional[Dict[str, Any]]:  # <-- Quitar comillas
        if self.path.exists():
            checkpoint = torch.load(self.path)
            visualizer.all_metrics = checkpoint.get('all_metrics', [])
            logger.info(f"Checkpoint cargado: {len(checkpoint['completed_variants'])} variantes completadas")
            return checkpoint['best_params']
        return None
    
    # En CheckpointManager
    def save_checkpoint(self, best_params: Dict, completed_variants: List[str], 
                    all_metrics: List[Dict]):
        checkpoint = {
            'best_params': best_params,
            'completed_variants': completed_variants,
            'all_metrics': all_metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Guardar versión incremental
        torch.save(checkpoint, self.path)
        logger.info(f"Checkpoint guardado en {self.path}")
        
        # Backup cada 5 variantes
        if len(completed_variants) % 5 == 0:
            backup_path = self.path.parent / f"backup_{self.path.name}"
            torch.save(checkpoint, backup_path)


class ECGExperimentRunner:
    """Ejecuta el experimento completo de ECG"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_factory = ModelFactory(config)
        self.data_handler = DataHandler(config)
        self.visualizer = ResultsVisualizer()
        self.evaluator = ModelEvaluator(self.device)
        
        self.data_loaders = self.data_handler.load_data()
        self.full_dataset = ConcatDataset([
            self.data_loaders['train'].dataset,
            self.data_loaders['val'].dataset
        ])

    def run_experiment(self):
        """Flujo principal del experimento"""
        logger.info("=== Inicio de experimento ECG ===")
        setup = ExperimentSetup()
        run_dir = setup.create_run_directory("run")
        self.visualizer.current_run_dir = run_dir
        
        checkpoint = CheckpointManager(run_dir / 'checkpoint.pth')
        best_params = checkpoint.load_checkpoint(self.visualizer)

        if not best_params:
            best_params = self._run_hyperparameter_optimization()
            checkpoint.save_checkpoint(best_params, [], self.visualizer.all_metrics)

        self._process_variants(best_params, checkpoint)
        self.visualizer.generate_comparative_report()
        self._cleanup_resources()

    
    # ECGExperimentRunner.py  (dentro de la clase)
    # ------------------------------------------------------------------
    def _run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """
        Lanza OptunaOptimizer o RayOptimizer según config.hpo_backend.
        """
        backend = self.config.hpo_backend.lower()
        if backend == "optuna":
            from src.optimization.optuna_optimizer import OptunaOptimizer
            optimizer = OptunaOptimizer(
                config=self.config,
                data_loaders=[self.data_loaders['train'], self.data_loaders['val']],
                model_factory=self.model_factory
            )
        elif backend == "ray":
            from src.optimization.ray_optimizer import RayOptimizer
            optimizer = RayOptimizer(
                config=self.config,
                data_loaders=[self.data_loaders['train'], self.data_loaders['val']],
                model_factory=self.model_factory
            )
        else:
            raise ValueError(f"HPO backend «{backend}» no soportado (usa 'optuna' o 'ray').")

        return optimizer.optimize(list(self.config.variants))


    # ───────────────────────── _process_variants ─────────────────────────
    # ───────────────────────── _process_variants ─────────────────────────
    def _process_variants(self, best_params: Dict[str, Any],
                          checkpoint: CheckpointManager):
        """
        1) Toma la mejor configuración de cada variante.
        2) Separa el learning-rate.
        3) Elimina claves meta como 'Best Trial' antes de crear el modelo.
        4) Guarda métricas y checkpoint incremental.
        """
        completed_variants = []

        for variant in self.config.variants:
            if variant in completed_variants:
                continue

            logger.info(f"Procesando variante: {variant}")

            # 1) Config ganadora
            best_cfg = best_params[variant]

            # 2) LR aparte
            lr_key = f"lr_{variant}"
            lr = best_cfg[lr_key]

            # 3) Hiper-parámetros reales
            params = {
                k: v
                for k, v in best_cfg.items()
                if k not in (lr_key, "Best Trial")
            }

            # 4) Entrenar, evaluar y guardar
            metrics, model = self._run_cross_validation(variant, params, lr)
            self._save_results(metrics, model, variant, best_cfg)

            completed_variants.append(variant)
            checkpoint.save_checkpoint(best_params,
                                        completed_variants,
                                        self.visualizer.all_metrics)



    def _run_cross_validation(self, variant: str, params: Dict[str, Any], lr: float) -> Tuple[Dict[str, Any], torch.nn.Module]:
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.full_dataset), 1):
            logger.info(f"Fold {fold}/{self.config.n_splits}")
            train_ld, val_ld = self._create_fold_loaders(train_idx, val_idx)
            fold_metric, model = self._train_fold(variant, params, lr, train_ld, val_ld)
            fold_metrics.append(fold_metric)
        
        return self._aggregate_metrics(fold_metrics), self._train_final_model(variant, params, lr)

    def _create_fold_loaders(self, train_idx: List[int], val_idx: List[int]) -> Tuple[DataLoader, DataLoader]:
        train_sub = Subset(self.full_dataset, train_idx)
        val_sub = Subset(self.full_dataset, val_idx)
        return (
            DataLoader(train_sub, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_sub, batch_size=self.config.batch_size, shuffle=False)
        )

    def _train_fold(self, variant: str, params: Dict[str, Any], lr: float, 
                   train_ld: DataLoader, val_ld: DataLoader) -> Tuple[Dict[str, float], torch.nn.Module]:
        model = self.model_factory.create_model(variant, params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = TrainingEngine(model, optimizer, self.device, torch.nn.MSELoss())
        train_engine, val_engine = trainer.create_engines()
        
        train_losses, val_losses = self._train_model(train_engine, val_engine, train_ld, val_ld)
        metrics = self.evaluator.evaluate_model(model, val_ld)
        
        return {**metrics, 'avg_train_loss': train_losses, 'avg_val_loss': val_losses}, model

    def _train_model(self, train_engine, val_engine, train_ld: DataLoader, val_ld: DataLoader) -> Tuple[List[float], List[float]]:
        train_losses, val_losses = [], []
        
        @train_engine.on(Events.EPOCH_COMPLETED)
        def log_epoch(engine):
            torch.cuda.empty_cache()
            train_loss = engine.state.output['loss']
            train_losses.append(train_loss)
            
            val_engine.run(val_ld)
            val_losses.append(val_engine.state.metrics['loss'])
            logger.debug(f"Epoch {len(train_losses)} - Train: {train_loss:.4f}, Val: {val_losses[-1]:.4f}")

        train_engine.run(train_ld, max_epochs=self.config.final_epochs)
        return train_losses, val_losses

    def _aggregate_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agrega métricas de K-Fold correctamente"""
        aggregated = {}
        # Procesar métricas principales (MAE, RMSE, MSE, CC)
        base_metrics = ['MAE', 'RMSE', 'MSE', 'CC']
        for metric in base_metrics:
            # Extraer valores de cada fold
            fold_means = [m[f"{metric}_mean"] for m in fold_metrics]
            fold_stds = [m[f"{metric}_std"] for m in fold_metrics]

            # Calcular estadísticas agregadas
            aggregated[f"{metric}_mean"] = np.mean(fold_means)  # Promedio entre folds
            aggregated[f"{metric}_std"] = np.std(fold_means)    # Std entre folds
            aggregated[f"{metric}_inter_std"] = np.mean(fold_stds)  # Std inter-electrodo
        
        # Añadir estadísticas del modelo (son constantes entre folds)
        model_stats = ['num_params', 'FLOPs', 'inference_latency_ms']
        for stat in model_stats:
            if stat in fold_metrics[0]:
                aggregated[stat] = fold_metrics[0][stat]
        
        return aggregated


    def _train_final_model(self, variant: str, params: Dict[str, Any], lr: float) -> torch.nn.Module:
        logger.info("Entrenando modelo final")
        full_loader = DataLoader(self.full_dataset, batch_size=self.config.batch_size, shuffle=True)
        model = self.model_factory.create_model(variant, params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = TrainingEngine(model, optimizer, self.device, torch.nn.MSELoss())
        trainer.create_engines()[0].run(full_loader, max_epochs=self.config.final_epochs)
        return model

    def _save_results(self, metrics: Dict[str, Any], model: torch.nn.Module, 
                     variant: str, best_params: Dict[str, Any]):
        stats = self.evaluator.compute_model_stats(model, self.config)
        preds, targets = self.evaluator.collect_predictions(model, self.data_loaders['val'])
        
        self.visualizer.save_results(
            results={**metrics, **stats, "Best Trial": self._format_best_trial(best_params)},
            variant=variant,
            model=model.cpu(),
            preds=preds,
            targets=targets,
            dataset_name=Path(self.config.data_path).stem
        )
        
        del model, preds, targets
        torch.cuda.empty_cache()

    def _format_best_trial(self, params: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""

    def _cleanup_resources(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Memoria GPU liberada: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

def main():
    # ───────────────── Configuración de logging ──────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # ─────────── Añadimos project_root al PYTHONPATH global ───────
    os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

    # ────────────── Inicialización (única) de Ray ─────────────────
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # ───────────────────── Lista de datasets ──────────────────────
    datasets = ["kitpvc", "utah2002", "bordeaux", "auckland"]

    try:
        for ds in datasets:
            logger.info(f"=== Processing dataset: {ds} ===")
            cfg = TrainingConfig(
                data_path=project_root / "data" / ds / "raw" / f"{ds}.mat",
                n_splits=5
            )
            ECGExperimentRunner(cfg).run_experiment()
            torch.cuda.empty_cache()
            logger.info(f"=== Dataset {ds} completed ===\n")

    except Exception as e:
        logger.critical(f"Main execution error: {e}")
        sys.exit(1)

    finally:
        # ──────────────── Apagamos Ray al final ───────────────────
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()