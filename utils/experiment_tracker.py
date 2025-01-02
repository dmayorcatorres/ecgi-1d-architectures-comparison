import os
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name="final_experiment"):
        """
        Inicializa el rastreador de experimentos con un nombre único.

        Args:
            experiment_name (str, opcional): Nombre del experimento. Si no se proporciona, se usa uno predeterminado.
        """
        # Usamos un nombre fijo para la carpeta, para que no se cree una nueva carpeta cada vez.
        self.experiment_name = experiment_name
        self.base_path = os.path.join("./experiments", self.experiment_name)  # Carpeta común para todos los experimentos
        self.models_path = os.path.join(self.base_path, "models")
        self.logs_path = os.path.join(self.base_path, "logs")
        self.plots_path = os.path.join(self.base_path, "plots")
        self.metrics_file = os.path.join(self.base_path, "metrics.json")
        self.config_file = os.path.join(self.base_path, "config.json")

        self._create_directories()
        self._initialize_file(self.metrics_file)
        self._initialize_file(self.config_file)

    def _create_directories(self):
        """Crea los directorios necesarios para el experimento."""
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def _initialize_file(self, file_path):
        """Inicializa un archivo vacío si no existe."""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)

    def get_model_path(self):
        """Devuelve la ruta donde se guardan los modelos."""
        return self.models_path

    def get_log_path(self):
        """Devuelve la ruta donde se guardan los registros."""
        return self.logs_path

    def get_plot_path(self):
        """Devuelve la ruta donde se guardan los gráficos."""
        return self.plots_path

    def get_experiment_path(self):
        """Devuelve la ruta base del experimento."""
        return self.base_path

    def log_experiment_info(self, logger):
        """Registra la información del experimento en los logs."""
        logger.info(f"📂 Experimento registrado en: {self.base_path}")
        logger.info(f"📁 Modelos: {self.models_path}")
        logger.info(f"📄 Logs: {self.logs_path}")
        logger.info(f"📊 Gráficas: {self.plots_path}")
        logger.info(f"📝 Métricas: {self.metrics_file}")
        logger.info(f"⚙️ Configuración: {self.config_file}")

    def log_config(self, config_dict):
        """Guarda la configuración del experimento en `config.json`."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print("⚙️ Configuración registrada correctamente en config.json.")
        except Exception as e:
            print(f"❌ Error al guardar la configuración: {e}")

    def add_experiment_note(self, note):
        """Agrega una nota al archivo `metrics.json`."""
        try:
            with open(self.metrics_file, 'r+') as f:
                metrics = json.load(f)
                if "notes" not in metrics:
                    metrics["notes"] = []
                metrics["notes"].append(note)
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print("📝 Nota agregada correctamente al experimento.")
        except Exception as e:
            print(f"❌ Error al agregar la nota: {e}")

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
                metrics['summary'] = summary_dict
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print("📊 Resumen del experimento registrado correctamente.")
        except Exception as e:
            print(f"❌ Error al registrar el resumen: {e}")
