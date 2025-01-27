# src/__init__.py
import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Archivo de configuración no encontrado en: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)