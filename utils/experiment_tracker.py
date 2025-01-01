import os
import json
from datetime import datetime


class ExperimentTracker:
    def __init__(self, experiment_name=None):
        """
        Inicializa el rastreador de experimentos con un nombre único.
        
        Args:
            experiment_name (str): Nombre del experimento. Si no se proporciona, se genera automáticamente.
        """
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.base_path = os.path.join("./experiments", experiment_name)
        self.models_path = os.path.join(self.base_path, "models")
        self.logs_path = os.path.join(self.base_path, "logs")
        self.plots_path = os.path.join(self.base_path, "plots")
        self.metrics_file = os.path.join(self.base_path, "metrics.json")
        self.config_file = os.path.join(self.base_path, "config.json")

        # Crear directorios si no existen
        self._create_directories()
        
        # Inicializar archivo de métricas si no existe
        self._initialize_file(self.metrics_file)
        
        # Inicializar archivo de configuración si no existe
        self._initialize_file(self.config_file)

    def _create_directories(self):
        """Crea los directorios necesarios para el experimento."""
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def _initialize_file(self, file_path):
        """Inicializa un archivo con un diccionario vacío si no existe."""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)

    def get_model_path(self):
        return self.models_path

    def get_log_path(self):
        return self.logs_path

    def get_plot_path(self):
        return self.plots_path

    def get_experiment_path(self):
        return self.base_path

    def log_experiment_info(self, logger):
        """
        Registra información general del experimento en los logs.
        
        Args:
            logger (logging.Logger): Instancia del logger.
        """
        logger.info(f"📂 Experimento registrado en: {self.base_path}")
        logger.info(f"📁 Modelos: {self.models_path}")
        logger.info(f"📄 Logs: {self.logs_path}")
        logger.info(f"📊 Plots: {self.plots_path}")
        logger.info(f"📝 Métricas: {self.metrics_file}")
        logger.info(f"⚙️ Configuración: {self.config_file}")

    def log_metric(self, metric_name, metric_value, step=None):
        """
        Registra una métrica en el archivo `metrics.json`.

        Args:
            metric_name (str): Nombre de la métrica.
            metric_value (float): Valor de la métrica.
            step (int, opcional): Paso (epoch) asociado a la métrica.
        """
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

    def log_config(self, config_dict):
        """
        Guarda la configuración del experimento en `config.json`.

        Args:
            config_dict (dict): Configuración del experimento.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print("⚙️ Configuración registrada correctamente en config.json.")
        except Exception as e:
            print(f"❌ Error al guardar la configuración: {e}")

    def add_experiment_note(self, note):
        """
        Agrega una nota al archivo `metrics.json`.

        Args:
            note (str): Nota sobre el experimento.
        """
        try:
            with open(self.metrics_file, 'r+') as f:
                metrics = json.load(f)
                if "notes" not in metrics:
                    metrics["notes"] = []
                metrics["notes"].append(note)
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print("📝 Nota añadida al experimento.")
        except Exception as e:
            print(f"❌ Error al agregar nota: {e}")
