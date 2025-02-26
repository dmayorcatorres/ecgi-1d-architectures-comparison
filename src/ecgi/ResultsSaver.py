import os
import datetime
import matplotlib.pyplot as plt
import logging
import pandas as pd
import torch

# Configuración básica de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

###############################################################################
# ResultsSaver: Maneja el guardado de resultados, figuras, modelos y tensores.
###############################################################################
class ResultsSaver:
    """
    Handles saving experiment results, figures, models, and tensors.

    Every time an experiment is executed, a folder is created under the provided
    base directory (e.g., "experiment_results/cnns") with the current date and time,
    where final results (CSV table), visualization images, model checkpoints, and tensors
    are stored.
    """
    def __init__(self, base_dir: str):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # Create a subfolder with timestamp (YYYYMMDD_HHMMSS)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        logger.info(f"Results folder created: {self.run_dir}")
    
    def save_csv(self, data: pd.DataFrame, filename: str = "final_results.csv") -> None:
        path = os.path.join(self.run_dir, filename)
        data.to_csv(path, index=False)
        logger.info(f"CSV results saved to {path}")
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self.run_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved to {path}")
    
    def save_model(self, model: torch.nn.Module, filename: str = "model.pt") -> None:
        path = os.path.join(self.run_dir, filename)
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def save_tensor(self, tensor: torch.Tensor, filename: str = "tensor.pt") -> None:
        path = os.path.join(self.run_dir, filename)
        torch.save(tensor, path)
        logger.info(f"Tensor saved to {path}")
    
    def save_architecture_graph(self, dot, filename: str):
        graph_dir = os.path.join(self.run_dir, "architecture_graphs")
        os.makedirs(graph_dir, exist_ok=True)
        dot_path = os.path.join(graph_dir, f"{filename}.dot")
        dot.save(dot_path)
        logger.info(f"Architecture graph saved to {dot_path}")