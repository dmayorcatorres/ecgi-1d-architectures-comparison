import os
import json
import datetime

class ExperimentTracker:
    """
    Clase para gestionar métricas y registros de un experimento.
    """

    def __init__(self, config_gan=None, config_paths=None):
        """
        Constructor del experiment tracker.

        Args:
            config_gan (dict): Configuraciones relacionadas con la GAN.
            config_paths (dict): Rutas de archivos/carpetas para logs, dataset, etc.
        """
        self.config_gan = config_gan
        self.config_paths = config_paths
        
        # Directorio donde se guardarán los registros
        self.experiment_dir = os.path.join(
            config_paths.get("LOG_PATH", "./logs"),
            f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Archivo para almacenar métricas
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        # Inicializar el archivo de métricas
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump({}, f, indent=4)

        # Historial de pérdidas
        self.loss_history = []

    def log_metric(self, metric_name, metric_value, step=None):
        """Registra una métrica en el archivo `metrics.json`."""
        try:
            with open(self.metrics_file, 'r+') as f:
                metrics = json.load(f)
                if step is not None:
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append({"step": step, "value": metric_value})
                else:
                    metrics[metric_name] = metric_value
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print(f"📊 Métrica registrada: {metric_name} = {metric_value} (Step: {step})")
        except Exception as e:
            print(f"❌ Error al registrar métrica {metric_name}: {e}")

    def get_metrics(self):
        """Devuelve todas las métricas registradas en `metrics.json`."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error al leer métricas: {e}")
            return {}

    def log_summary(self, summary_dict):
        """Agrega un resumen general del experimento."""
        try:
            with open(self.metrics_file, 'r+') as f:
                metrics = json.load(f)
                for key, value in summary_dict.items():
                    metrics[key] = value
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print("📄 Resumen del experimento registrado correctamente.")
        except Exception as e:
            print(f"❌ Error al registrar resumen de experimento: {e}")

    def append_loss(self, loss_value):
        """Agrega una nueva pérdida al historial interno."""
        self.loss_history.append(loss_value)

    def get_final_loss(self):
        """Devuelve la última pérdida registrada en el historial."""
        if self.loss_history:
            return self.loss_history[-1]
        return float('inf')