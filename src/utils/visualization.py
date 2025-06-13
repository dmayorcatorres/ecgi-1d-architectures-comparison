import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
from src.ecgi.ECGiVisualizer import ECGiVisualizer
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The behavior of DataFrame concatenation")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Visualiza y guarda resultados del experimento sin renombrar claves de métricas."""
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.current_run_dir: Optional[Path] = None
        self.all_metrics: List[Dict[str, Any]] = []

        # Orden final de columnas, usando las claves tal cual vienen de _compute_metrics y de compute_model_stats
        self.metric_config = {
            'columns': [
                "Run Folder", "Dataset", "Variant",
                # Time-domain metrics
                "MAE_mean", "MAE_std",
                "RMSE_mean", "RMSE_std",
                "MSE_mean", "MSE_std",
                "CC_mean", "CC_std",
                # Inter-electrode Std (si se añade con esas claves)
                "MAE_inter_std", "RMSE_inter_std",
                "MSE_inter_std", "CC_inter_std",
                # Estadísticas del modelo
                "num_params", "FLOPs", "inference_latency_ms",
                "Model Size (MB)",
                # Hiperparámetros
                "Best Trial",
            ]
        }

    def save_results(self,
                     results: Dict[str, Any],
                     variant: str,
                     model: nn.Module,
                     preds: torch.Tensor,
                     targets: torch.Tensor,
                     dataset_name: str):
        """
        1) Crea carpeta de variante
        2) Guarda modelo y tensores
        3) Calcula tamaño del modelo
        4) Añade metadatos + métricas en self.all_metrics
        5) Guarda en Excel y genera gráficas
        """
        # 1) Carpeta de la variante
        variant_dir = self.current_run_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        # 2) Guardar modelo y tensores
        model_path = variant_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        torch.save({'preds': preds, 'targets': targets}, variant_dir / "data.pt")

        # 3) Calcular tamaño del modelo
        model_size_mb = round(model_path.stat().st_size / 2**20, 2)
        results["Model Size (MB)"] = model_size_mb  # <-- Clave exacta

        # 4) Construir el registro completo
        record = {
            "Run Folder": self.current_run_dir.name,
            "Dataset": dataset_name,
            "Variant": variant,
            **results  # Incluye todas las claves generadas
        }
        self.all_metrics.append(record)

        # 5) Guardar en el Excel global (una hoja por variante)
        df = pd.DataFrame([record])
        excel_path = self.current_run_dir / f"metrics_{dataset_name}.xlsx"
        with pd.ExcelWriter(excel_path,
                            mode='a' if excel_path.exists() else 'w',
                            engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=variant, index=False)

        # 6) Generar gráficas para esta variante
        preds_np   = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        self._save_plots(results, variant_dir, preds_np, targets_np)

    def _save_plots(self, results: Dict[str, Any], variant_dir: Path,
                   preds: np.ndarray, targets: np.ndarray):
        if 'avg_train_loss' in results and 'avg_val_loss' in results:
            self._plot_loss_curves(results['avg_train_loss'],
                                   results['avg_val_loss'],
                                   variant_dir)
        self._plot_electrode_comparison(preds, targets, variant_dir)

    def _plot_loss_curves(self, train_loss: List[float],
                          val_loss: List[float],
                          output_dir: Path):
        plt.figure(figsize=(8, 4))
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png")
        plt.close()

    def _plot_electrode_comparison(self, preds: np.ndarray,
                                   targets: np.ndarray,
                                   output_dir: Path):
        try:
            ECGiVisualizer.plot_electrodes(
                original_data_3d=targets,
                reconstructed_data_3d=preds,
                fs=1000,
                signal_label="ECG",
                num_to_plot=5,
                save_path=str(output_dir / "electrode_comparison.png")
            )
        except Exception as e:
            logger.error(f"Error generando comparación de electrodos: {e}")

    def generate_comparative_report(self):
        """
        1) Prepara DataFrame con self.all_metrics
        2) Convierte las columnas numéricas
        3) Rellena y ordena columnas según metric_config['columns']
        4) Guarda summary, genera gráficas y actualiza acumulativo global
        """
        if not self.current_run_dir:
            raise ValueError("Directorio no configurado")

        # 1) DataFrame completo
        df = pd.DataFrame(self.all_metrics)

        # 2) Convertir columnas numéricas
        numeric_cols = [
            c for c in df.columns 
            if any(k in c for k in ['_mean', '_std', 'num_params', 'FLOPs', 'inference_latency_ms', 'Model Size (MB)'])
        ]
        
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 3) Rellenar cualquier columna faltante y ordenar
        for col in self.metric_config['columns']:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"Columna faltante: {col} – rellenando con NaN")
        df = df[self.metric_config['columns']]

        # 4) Guardar el summary
        df.to_csv(self.current_run_dir / "summary_metrics.csv", index=False)

        # 5) Gráficas comparativas
        self._generate_plots(df)

        # 6) Actualizar acumulativo global
        self._update_global_accumulative()

    def _generate_plots(self, df: pd.DataFrame):
        for metric in ['MAE_mean', 'MSE_mean', 'CC_mean']:
            if metric in df.columns:
                self._create_bar_plot(df, metric)
        self._create_comparative_plot(df)

    def _create_bar_plot(self, df: pd.DataFrame, metric: str):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df))
        err_col = metric.replace('_mean', '_std')
        yerr = df.get(err_col)
        plt.bar(x, df[metric], yerr=yerr, capsize=4, alpha=0.7)
        plt.xticks(x, df['Variant'], rotation=45)
        plt.title(f"{metric} ± {err_col}")
        plt.tight_layout()
        plt.savefig(self.current_run_dir / f"{metric}_plot.png")
        plt.close()

    def _create_comparative_plot(self, df: pd.DataFrame):
        norm_cols = self._normalize_metrics(df)
        if not norm_cols:
            return
        plt.figure(figsize=(12, 8))
        width = 0.8 / len(norm_cols)
        x = np.arange(len(df))
        for i, col in enumerate(norm_cols):
            plt.bar(x + i*width - 0.4 + width/2, df[col], width=width, label=col)
        plt.xticks(x, df['Variant'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.current_run_dir / "comparative_metrics.png")
        plt.close()

    def _normalize_metrics(self, df: pd.DataFrame) -> List[str]:
        normalized_cols: List[str] = []
        for metric in ['MAE_mean', 'RMSE_mean', 'CC_mean', 'MSE_mean']:
            if metric not in df.columns:
                continue
            if metric == 'CC_mean':
                df[f"{metric}_norm"] = df[metric] / df[metric].max()
            else:
                mn, mx = df[metric].min(), df[metric].max()
                denom = (mx - mn) or 1e-8
                df[f"{metric}_norm"] = 1 - (df[metric] - mn) / denom
            normalized_cols.append(f"{metric}_norm")
        return normalized_cols

    def _update_global_accumulative(self):
        accum_path = self.base_dir / "00_accumulative_filtered_results.xlsx"
        df = pd.DataFrame(self.all_metrics)
        if accum_path.exists():
            existing = pd.read_excel(accum_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_excel(accum_path, index=False)
