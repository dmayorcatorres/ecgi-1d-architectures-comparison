import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, Any
from src.ecgi.ECGiVisualizer import ECGiVisualizer

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    def __init__(self, base_dir="experiment_results"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.all_metrics: list[Dict[str, Any]] = []
        self.current_run_dir: str | None = None
        self.dataset_name: str = ""  # Nueva variable de instancia

    def create_run_directory(self, experiment_name: str) -> str:
        """Crea directorio único para cada ejecución con timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        return self.current_run_dir

    def save_results(self, results: Dict[str, Any], variant: str, model: torch.nn.Module,
                     preds: torch.Tensor | None = None, targets: torch.Tensor | None = None):
        try:
            assert self.current_run_dir is not None, "Run directory not set"
            variant_dir = os.path.join(self.current_run_dir, variant)
            os.makedirs(variant_dir, exist_ok=True)

            model_path = os.path.join(variant_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            results['Model Size (MB)'] = round(os.path.getsize(model_path) / (1024 * 1024), 2)

            run_folder_name = os.path.basename(self.current_run_dir)

            results_renamed = {
                'Run Folder': run_folder_name,  # Nueva columna
                'Dataset': self.dataset_name,    # Nueva columna
                'Variant': variant,
                'MAE (Mean)': results.get('MAE_mean_mean'),
                'MAE (KFold Std)': results.get('MAE_mean_std'),
                'MAE (Inter-electrode Std)': results.get('MAE_std_mean'),
                'RMSE (Mean)': results.get('RMSE_mean_mean'),
                'RMSE (KFold Std)': results.get('RMSE_mean_std'),
                'RMSE (Inter-electrode Std)': results.get('RMSE_std_mean'),
                'CC (Mean)': results.get('CC_mean_mean'),
                'CC (KFold Std)': results.get('CC_mean_std'),
                'CC (Inter-electrode Std)': results.get('CC_std_mean'),
                '#Parameters': results.get('num_params'),
                'FLOPs': results.get('FLOPs'),
                'Inference Latency (ms)': results.get('inference_latency_ms'),
                'Model Size (MB)': results.get('Model Size (MB)')
            }

            pd.DataFrame([results_renamed]).to_csv(
                os.path.join(variant_dir, "filtered_metrics.csv"), index=False
            )

            if preds is not None and targets is not None:
                try:
                    ECGiVisualizer.plot_electrodes(
                        original_data_3d=targets.numpy(),
                        reconstructed_data_3d=preds.numpy(),
                        fs=1000,
                        signal_label="ECG",
                        num_to_plot=5,
                        save_path=os.path.join(variant_dir, "electrode_comparison.png")
                    )
                except Exception:
                    logger.exception("Error generando visualización de electrodos para %s", variant)

            self.all_metrics.append(results_renamed)
        except Exception:
            logger.exception("Error guardando resultados para %s", variant)
            raise


    def generate_comparative_report(self):
        assert self.current_run_dir is not None, "Run directory not set"
        df = pd.DataFrame(self.all_metrics)
        df.to_csv(os.path.join(self.current_run_dir, "summary_metrics.csv"), index=False)

        for metric in ['MAE (Mean)', 'RMSE (Mean)', 'CC (Mean)']:
            std_col = metric.replace("(Mean)", "(KFold Std)")
            plt.figure(figsize=(10, 6))
            x = np.arange(len(df))
            plt.bar(x, df[metric], yerr=df[std_col], capsize=4)
            plt.xticks(x, df['Variant'], rotation=45)
            plt.ylabel(metric)
            plt.title(f"{metric} ± KFold Std")
            plt.tight_layout()
            plt.savefig(os.path.join(self.current_run_dir, f"{metric.replace(' ', '_')}_barplot.png"))
            plt.close()

        self._update_global_accumulative(df)
    def _update_global_accumulative(self, new_data: pd.DataFrame):
        accum_path = os.path.join(self.base_dir, "00_accumulative_filtered_results.xlsx")
        if os.path.exists(accum_path):
            existing = pd.read_excel(accum_path)
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
        combined.to_excel(accum_path, index=False)
