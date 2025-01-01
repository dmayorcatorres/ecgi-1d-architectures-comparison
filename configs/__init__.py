import os
import yaml

# Cargar la configuración base
BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'base.yaml')

with open(BASE_CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
