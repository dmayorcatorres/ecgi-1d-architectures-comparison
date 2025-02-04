import yaml
import os

def load_config():
    """
    Carga y fusiona la configuración general y la de arquitectura.
    """
    config_general_path = os.path.join(os.path.dirname(__file__), "config_general.yml")
    config_architecture_path = os.path.join(os.path.dirname(__file__), "config_architecture.yml")

    with open(config_general_path, "r") as f:
        config_general = yaml.safe_load(f)

    with open(config_architecture_path, "r") as f:
        config_architecture = yaml.safe_load(f)

    # Fusionar configuraciones
    config = {**config_general, **config_architecture}
    return config
