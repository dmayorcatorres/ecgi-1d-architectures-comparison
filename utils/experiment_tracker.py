import os
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name=None):
        """
        Initializes the experiment tracker with a unique name.

        Args:
            experiment_name (str): Name of the experiment. If not provided, it is generated automatically.
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_path = os.path.join("./experiments", self.experiment_name)
        self.models_path = os.path.join(self.base_path, "models")
        self.logs_path = os.path.join(self.base_path, "logs")
        self.plots_path = os.path.join(self.base_path, "plots")
        self.metrics_file = os.path.join(self.base_path, "metrics.json")
        self.config_file = os.path.join(self.base_path, "config.json")

        self._create_directories()
        self._initialize_file(self.metrics_file)
        self._initialize_file(self.config_file)

    def _create_directories(self):
        """Creates the necessary directories for the experiment."""
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def _initialize_file(self, file_path):
        """Initializes a file with an empty dictionary if it does not exist."""
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
        Logs general experiment information.

        Args:
            logger (logging.Logger): Logger instance.
        """
        logger.info(f"📂 Experiment registered at: {self.base_path}")
        logger.info(f"📁 Models: {self.models_path}")
        logger.info(f"📄 Logs: {self.logs_path}")
        logger.info(f"📊 Plots: {self.plots_path}")
        logger.info(f"📝 Metrics: {self.metrics_file}")
        logger.info(f"⚙️ Configuration: {self.config_file}")

    def log_metric(self, metric_name, metric_value, step=None):
        """
        Logs a metric in the `metrics.json` file.

        Args:
            metric_name (str): Name of the metric.
            metric_value (float): Value of the metric.
            step (int, optional): Step (epoch) associated with the metric.
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

            print(f"📊 Metric logged: {metric_name} = {metric_value} (Step: {step})")
        except Exception as e:
            print(f"❌ Error logging metric {metric_name}: {e}")

    def log_config(self, config_dict):
        """
        Saves the experiment configuration in `config.json`.

        Args:
            config_dict (dict): Experiment configuration.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print("⚙️ Configuration successfully logged in config.json.")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")

    def add_experiment_note(self, note):
        """
        Adds a note to the `metrics.json` file.

        Args:
            note (str): Note about the experiment.
        """
        try:
            with open(self.metrics_file, 'r+') as f:
                metrics = json.load(f)
                if "notes" not in metrics:
                    metrics["notes"] = []
                metrics["notes"].append(note)
                f.seek(0)
                json.dump(metrics, f, indent=4)
            print("📝 Note added to the experiment.")
        except Exception as e:
            print(f"❌ Error adding note: {e}")