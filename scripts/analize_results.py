# scripts/analyze_results.py

import os
import sys
import logging
import time
import yaml
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader
from collections import defaultdict

import torch
import numpy as np

# Configuración de paths
_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR_PATH, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.configs.base_config import TrainingConfig
from src.data.datasets import ECGDataLoaderFactory
from src.models.model_factory import ModelFactory
from src.utils.visualization import ResultsVisualizer
from src.ecgi.ECGMetricsComparator import ECGMetricsComparator

logger = logging.getLogger(__name__)


class ECGResultsAnalyzer:
    def __init__(self, config: TrainingConfig, run_dir: str):
        self.config = config
        self.run_dir = run_dir
        self.device = self._setup_device()
        self.val_loader = self._prepare_data()
        self.model_factory = ModelFactory(self.config)
        self.visualizer = ResultsVisualizer()

    def _setup_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Analyzer] Using device: {device}")
        return device

    def _prepare_data(self) -> DataLoader:
        factory = ECGDataLoaderFactory(self.config)
        _, val_loader = factory.create_loaders()
        return val_loader

    def _load_checkpoint(self, path: str) -> Tuple[Dict, Dict]:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            if "state_dict" in ckpt and "hyper_params" in ckpt:
                return ckpt["state_dict"], ckpt["hyper_params"]
            return ckpt, {}
        except Exception as e:
            logger.error(f"Error loading checkpoint {path}: {str(e)}")
            raise

    def _safe_load_model(self, state_dict: Dict, params: Dict, variant: str) -> torch.nn.Module:
        model = self.model_factory.create_model(variant, params).to(self.device)
        
        # Filtrar claves no esenciales
        filtered_dict = {k: v for k, v in state_dict.items() 
                       if "total_ops" not in k and "total_params" not in k}
        
        # Carga de parámetros con manejo de errores
        try:
            model.load_state_dict(filtered_dict, strict=False)
        except RuntimeError as e:
            logger.warning(f"Partial weights loading: {str(e)}")
        
        return model

    def _compute_model_metrics(self, model: torch.nn.Module) -> Dict[str, Any]:
        stats = {
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "inference_latency": self._measure_inference_time(model)
        }
        
        try:
            from thop import profile
            dummy_input = torch.randn(1, self.config.input_channels, self.config.time_steps).to(self.device)
            flops, macs = profile(model, inputs=(dummy_input,), verbose=False)
            stats.update({"FLOPs": flops, "MACs": macs})
        except ImportError:
            logger.warning("THOP not installed, skipping FLOPs/MACs calculation")
        
        return stats

    def _measure_inference_time(self, model: torch.nn.Module) -> float:
        dummy = torch.randn(1, self.config.input_channels, self.config.time_steps).to(self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)
        
        # Medición real
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        return (time.perf_counter() - start) * 1000  # ms

    def _collect_predictions(self, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = model(inputs.to(self.device)).cpu()
                preds.append(outputs)
                targets.append(labels)
        
        return torch.cat(preds), torch.cat(targets)

    def _compute_metrics(self, targets: torch.Tensor, preds: torch.Tensor) -> Dict[str, float]:
        comparator = ECGMetricsComparator(fs=1000)
        comparator.set_time_domain_signals(targets, preds)
        
        metrics = {
            "MAE": comparator.compute_time_mae(),
            "CC": comparator.compute_time_cc(),
            "MSE": comparator.compute_time_mse_rmse()[0],
            "RMSE": comparator.compute_time_mse_rmse()[1],
            "SNR": comparator.compute_time_snr()
        }
        
        return {k: {"mean": v[0], "std": v[1]} for k, v in metrics.items()}

    def analyze(self):
        variants = [d for d in os.listdir(self.run_dir) 
                  if os.path.isdir(os.path.join(self.run_dir, d))]
        
        for variant in variants:
            var_path = os.path.join(self.run_dir, variant)
            ckpt_path = self._find_checkpoint(var_path)
            
            if not ckpt_path:
                logger.warning(f"No checkpoint found in {var_path}")
                continue
            
            try:
                state_dict, params = self._load_checkpoint(ckpt_path)
                model = self._safe_load_model(state_dict, params, variant)
                metrics = self._process_model(model, variant)
                self._save_results(model, metrics, variant)
            except Exception as e:
                logger.error(f"Error processing {variant}: {str(e)}")
                continue
            finally:
                torch.cuda.empty_cache()

        self.visualizer.generate_comparative_report() if variants else None

    def _find_checkpoint(self, path: str) -> str:
        for f in os.listdir(path):
            if f.endswith((".pt", ".pth")):
                return os.path.join(path, f)
        return None

    def _process_model(self, model: torch.nn.Module, variant: str) -> Dict:
        stats = self._compute_model_metrics(model)
        preds, targets = self._collect_predictions(model)
        metrics = self._compute_metrics(targets, preds)
        return {**stats, **metrics}

    def _save_results(self, model: torch.nn.Module, metrics: Dict, variant: str):
        self.visualizer.save_results(
            metrics_data=metrics,
            model_name=variant,
            model=model.cpu(),
            predictions=model.latest_predictions,
            targets=model.latest_targets
        )


def main():
    # Configuración
    RUN_DIR = os.path.join(_PROJECT_ROOT, "experiment_results/ECG_Experiment_20250420_060134")
    
    # Cargar configuración original del experimento
    config_path = os.path.join(RUN_DIR, "config.yaml")
    with open(config_path) as f:
        config = TrainingConfig(**yaml.safe_load(f))

    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("analysis.log")]
    )

    analyzer = ECGResultsAnalyzer(config, RUN_DIR)
    analyzer.analyze()


if __name__ == "__main__":
    main()