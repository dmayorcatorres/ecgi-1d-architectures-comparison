import os
import sys
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from ignite.engine import Events

# Configurar paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Importaciones locales
from src.configs.base_config import TrainingConfig
from src.data.datasets import ECGDataLoaderFactory
from src.models.model_factory import ModelFactory
from src.training.engines import TrainingEngine
from src.optimization.optuna_optimizer import OptunaOptimizer
from src.utils.visualization import ResultsVisualizer
from src.ecgi.ECGMetricsComparator import ECGMetricsComparator

# Configurar logging
logger = logging.getLogger(__name__)


class ECGExperimentRunner:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model_factory = ModelFactory(config)
        self.visualizer = ResultsVisualizer()

        # 1) Preparamos los datos
        self.data_loaders = self._prepare_data()

        # 2) Verificamos que inputs y targets tengan la misma longitud temporal
        self._verify_data_shapes(self.data_loaders['train'], name='train')
        self._verify_data_shapes(self.data_loaders['val'],   name='val')

        # 3) Creamos el dataset completo para CV y entrenamiento final
        self.full_dataset = self._create_full_dataset()

    def _setup_device(self) -> torch.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {device}")
        return device

    def _prepare_data(self) -> Dict[str, DataLoader]:
        logger.info("Inicializando cargadores de datos...")
        factory = ECGDataLoaderFactory(self.config)
        train_loader, val_loader = factory.create_loaders()
        # Actualizamos config basándonos en un batch de ejemplo
        self._update_config_with_sample_data(train_loader)
        return {'train': train_loader, 'val': val_loader}

    def _update_config_with_sample_data(self, train_loader: DataLoader):
        sample_ecg, sample_emg = next(iter(train_loader))
        self.config.input_channels = sample_ecg.size(1)
        self.config.output_channels = sample_emg.size(1)
        self.config.time_steps = sample_ecg.size(2)
        logger.debug(f"Dimensiones de entrada configuradas: {self.config.input_channels} canales, "
                     f"{self.config.time_steps} pasos temporales")


    def _verify_data_shapes(self,
                            loader: DataLoader,
                            name: str,
                            n_batches: int = 1) -> None:
        """
        Verifica en los primeros `n_batches` de `loader` que:
          - input.size(-1) == target.size(-1) == config.time_steps
          - input.size(1) == config.input_channels
          - target.size(1) == config.output_channels
        Lanza ValueError en caso de mismatch.
        """
        for idx, (inp, tgt) in enumerate(loader):
            t_in = inp.size(-1)
            t_tg = tgt.size(-1)
            c_in = inp.size(1)
            c_tg = tgt.size(1)
            if t_in != t_tg:
                raise ValueError(
                    f"[{name}] Batch {idx}: mismatch temporal "
                    f"in={t_in} vs tgt={t_tg}"
                )
            if t_in != self.config.time_steps:
                raise ValueError(
                    f"[{name}] Batch {idx}: pasos temporales "
                    f"esperados={self.config.time_steps}, pero got={t_in}"
                )
            if c_in != self.config.input_channels:
                raise ValueError(
                    f"[{name}] Batch {idx}: canales input "
                    f"esperados={self.config.input_channels}, pero got={c_in}"
                )
            if c_tg != self.config.output_channels:
                raise ValueError(
                    f"[{name}] Batch {idx}: canales target "
                    f"esperados={self.config.output_channels}, pero got={c_tg}"
                )
            if idx+1 >= n_batches:
                break
        logger.info(f"[{name}] Verificación de shapes OK en {idx+1} batch(es)")
        

    def _create_full_dataset(self) -> ConcatDataset:
        train_ds = self.data_loaders['train'].dataset
        val_ds   = self.data_loaders['val'].dataset
        return ConcatDataset([train_ds, val_ds])

    def _compute_model_stats(self, model: torch.nn.Module) -> Dict[str, Any]:
        stats = {
            'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2)
        }
        
        dummy = torch.randn(1, self.config.input_channels, self.config.time_steps).to(self.device)
        
        # Medición de latencia
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        stats['inference_latency_ms'] = (time.perf_counter() - start) * 1000

        # Cálculo de FLOPs
        stats.update(self._calculate_flops(model, dummy))
        
        logger.debug(f"Estadísticas del modelo: {stats}")
        return stats

    def _calculate_flops(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        try:
            from thop import profile
            flops, macs = profile(model, inputs=(input_tensor,), verbose=False)
            return {'FLOPs': flops, 'MACs': macs}
        except ImportError:
            logger.warning("thop no está instalado, no se pueden calcular FLOPs/MACs")
            return {'FLOPs': None, 'MACs': None}

    def _collect_predictions(self, model: torch.nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for batch_idx, (inp, tgt) in enumerate(loader):
                inp = inp.to(self.device)
                pred = model(inp).cpu()  # Mover a CPU inmediatamente
                all_preds.append(pred)
                all_tgts.append(tgt)
                
                # Liberar memoria del batch
                del inp, pred
                if batch_idx % 10 == 0:  # Cada 10 batches
                    torch.cuda.empty_cache()
                    logger.debug(f"Memoria durante inferencia: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        # Liberar memoria de listas intermedias
        torch.cuda.empty_cache()
        return torch.cat(all_preds), torch.cat(all_tgts)

    def _train_fold(self,
                    variant: str,
                    params: Dict[str, Any],
                    lr: float,
                    train_loader: DataLoader,
                    val_loader: DataLoader
                   ) -> Tuple[Optional[torch.nn.Module],
                              Dict[str, float],
                              List[float],
                              List[float]]:
        try:
            model = self.model_factory.create_model(variant, params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Usamos la firma legacy: (model, optimizer, device, criterion)
            trainer = TrainingEngine(model, optimizer, self.device, torch.nn.MSELoss())
            train_engine, val_engine = trainer.create_engines()

            train_losses: List[float] = []
            val_losses:   List[float] = []

            @train_engine.on(Events.EPOCH_COMPLETED)
            def log_epoch(engine):
                # 1) Limpiamos caché de GPU
                torch.cuda.empty_cache()

                # 2) Sacamos el float del dict
                train_loss = engine.state.output['loss']
                train_losses.append(train_loss)

                # 3) Ejecutamos validación
                val_engine.run(val_loader)
                val_loss = val_engine.state.metrics['loss']
                val_losses.append(val_loss)

                # 4) Debug con floats
                logger.debug(
                    f"Fold {len(train_losses)} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss:   {val_loss:.4f}"
                )

            # Entrenamos
            train_engine.run(train_loader, max_epochs=self.config.final_epochs)

            # Al terminar todos los epochs del fold, evaluamos métricas globales
            metrics = self._evaluate_model(model, val_loader)
            return model, metrics, train_losses, val_losses

        except Exception as e:
            logger.error(f"Error entrenando fold: {e}")
            return None, {}, [], []

        finally:
            # Limpieza exhaustiva
            if 'model' in locals():
                del model
                logger.debug("Modelo del fold liberado de memoria")
            if 'optimizer' in locals():
                del optimizer
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
            logger.debug(f"Memoria GPU después de fold: "
                         f"{torch.cuda.memory_allocated()/1024**2:.2f} MB")

    def _evaluate_model(self, model: torch.nn.Module, loader: DataLoader) -> Dict[str, float]:
        try:
            preds, tgts = self._collect_predictions(model, loader)
            return self.compute_all_metrics(tgts, preds)
        except Exception as e:
            logger.error(f"Error evaluando modelo: {str(e)}")
            return {}
        
    def compute_all_metrics(self, targets: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
        try:
            cmp = ECGMetricsComparator(fs=1000)
            cmp.set_time_domain_signals(targets, preds)
            return {
                'MAE_mean': cmp.compute_time_mae()[0],
                'MAE_std': cmp.compute_time_mae()[1],
                'CC_mean': cmp.compute_time_cc()[0],
                'CC_std': cmp.compute_time_cc()[1],
                'MSE_mean': cmp.compute_time_mse_rmse()[0][0],
                'MSE_std': cmp.compute_time_mse_rmse()[0][1],
                'RMSE_mean': cmp.compute_time_mse_rmse()[1][0],
                'RMSE_std': cmp.compute_time_mse_rmse()[1][1],
                'SNR_mean': cmp.compute_time_snr()[0],
                'SNR_std': cmp.compute_time_snr()[1]
            }
        except Exception as e:
            logger.error(f"Error calculando métricas: {str(e)}")
            return {}

    def _run_cross_validation(self, variant: str, params: Dict[str, Any], lr: float) -> Tuple[Dict[str, Any], torch.nn.Module]:
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        all_tr_losses, all_val_losses = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.full_dataset), 1):
            logger.info(f"Iniciando fold {fold}/{self.config.n_splits}")
            
            train_sub = Subset(self.full_dataset, train_idx)
            val_sub = Subset(self.full_dataset, val_idx)
            train_ld = DataLoader(train_sub, batch_size=self.config.batch_size, shuffle=True)
            val_ld = DataLoader(val_sub, batch_size=self.config.batch_size, shuffle=False)

            model, metrics, tl, vl = self._train_fold(variant, params, lr, train_ld, val_ld)
            if model:
                fold_metrics.append(metrics)
                all_tr_losses.append(tl)
                all_val_losses.append(vl)

        agg_metrics = self._aggregate_metrics(fold_metrics)
        avg_tr_loss = np.mean(all_tr_losses, axis=0).tolist()
        avg_val_loss = np.mean(all_val_losses, axis=0).tolist()
        
        final_model = self._train_final_model(variant, params, lr)
        
        return {**agg_metrics, 'avg_train_loss': avg_tr_loss, 'avg_val_loss': avg_val_loss}, final_model

    def _train_final_model(self, variant: str, params: Dict[str, Any], lr: float) -> torch.nn.Module:
        logger.info("Entrenando modelo final con todos los datos...")
        full_loader = DataLoader(self.full_dataset, batch_size=self.config.batch_size, shuffle=True)
        model = self.model_factory.create_model(variant, params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = TrainingEngine(model, optimizer, self.device, torch.nn.MSELoss())
        train_engine, _ = trainer.create_engines()
        train_engine.run(full_loader, max_epochs=self.config.final_epochs)
        return model

    def run_experiment(self):
        """Flujo principal del experimento"""
        logger.info("=== Inicio de experimento ECG ===")
        self.visualizer.create_run_directory("ECG_Experiment")

        try:
            # Fase 1: Optimización de hiperparámetros
            logger.info("Fase 1: Optimización de hiperparámetros")
            best_params = self._run_hyperparameter_optimization()

            # Fase 2: Validación cruzada y entrenamiento final
            logger.info("Fase 2: Validación cruzada y entrenamiento final")
            for variant in self.config.variants:
                logger.info(f"Procesando variante: {variant}")
                vp = best_params[variant].copy()
                lr = vp.pop(f"lr_{variant}")
                
                metrics, model = self._run_cross_validation(variant, vp, lr)
                self._save_results(metrics, model, variant)

            # Fase 3: Reporte final
            logger.info("Fase 3: Generación de reporte final")
            self.visualizer.generate_comparative_report()

        except Exception as e:
            logger.critical(f"Error crítico en el experimento: {str(e)}")
            raise
        finally:
            logger.info("=== Experimento completado ===")
            self._cleanup_resources()

    def _run_hyperparameter_optimization(self) -> Dict[str, Any]:
        opt = OptunaOptimizer(
            config=self.config,
            data_loaders=[self.data_loaders['train'], self.data_loaders['val']],
            model_factory=self.model_factory
        )
        best_params = opt.optimize(self.config.variants)
        
        # Limpiar después de optimización
        del opt
        torch.cuda.empty_cache()
        return best_params

    def _save_results(self, metrics: Dict[str, Any], model: torch.nn.Module, variant: str):

                # Extraer nombre del dataset del config
        dataset_name = os.path.basename(self.config.data_path).split('.')[0]
        self.visualizer.dataset_name = dataset_name  # Pasar al visualizador

        stats = self._compute_model_stats(model)
        preds, targets = self._collect_predictions(model, self.data_loaders['val'])
        
        # Mover datos a CPU antes de guardar
        self.visualizer.save_results(
            {**metrics, **stats}, 
            variant, 
            model=model.cpu(),  # Mover modelo a CPU
            preds=preds.cpu() if preds is not None else None,
            targets=targets.cpu() if targets is not None else None
        )
        
        # Limpieza explícita
        del model, preds, targets
        torch.cuda.empty_cache()
        logger.info(f"Memoria después de guardar resultados: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    def _cleanup_resources(self):
        """Limpieza final de recursos"""
        if torch.cuda.is_available():
            logger.info("Liberando memoria GPU residual")
            torch.cuda.empty_cache()
            logger.info(f"Memoria GPU después de limpieza: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def _aggregate_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calcula media y std de métricas a través de los folds"""
        aggregated = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
        return aggregated


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Lista de datasets a procesar
    datasets = [
        "kitpvc",
        "utah2002", 
        "bordeaux",
        "auckland"
    ]
    
    try:
        for dataset in datasets:
            logger.info(f"=== Procesando dataset: {dataset} ===")
            
            config = TrainingConfig(
                data_path=os.path.join(os.path.dirname(__file__), "..", "data", 
                                        dataset, "raw", f"{dataset}.mat"),
                n_splits=5
            )
            
            # Configurar visualizador para acumular resultados
            visualizer = ResultsVisualizer()
            visualizer.output_file = "resultados_comparativos.xlsx"  # Nombre único para todos
            
            runner = ECGExperimentRunner(config)
            runner.visualizer = visualizer  # Usar el mismo visualizador para todos
            runner.run_experiment()
            
            # Limpiar memoria entre datasets
            torch.cuda.empty_cache()
            logger.info(f"=== Dataset {dataset} completado ===\n")
   
    except Exception as e:
        logger.critical(f"Error en ejecución principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()