import sys
import os
import torch
import mlflow
import numpy as np
import random
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import optuna.visualization as vis
from functools import partial

from configs import load_config
sys.path.append(os.path.abspath(os.path.dirname(__file__)))



# Cargar el contenido de un único YAML
config = load_config("config.yml")

from data.data_loader import prepare_dataset
from training.trainer import initialize_models, train
from utils.logger import setup_logging
from utils.experiment_tracker import ExperimentTracker
from utils.early_stopping import EarlyStopping
from utils.saver import save_loss_plot, save_models

# Variables globales
logger = None
experiment_tracker = None
models = None

# Ajusta las claves requeridas si necesitas validación
REQUIRED_KEYS = ['DATA_PATH', 'DATASET', 'MODEL_PATH', 'LOG_PATH', 'RANGES']

def check_config_keys(required_keys, cfg):
    for k in required_keys:
        if k not in cfg:
            raise KeyError(f"Falta la clave '{k}' en config.")

def initialize_experiment(experiment_name):
    global logger
    log_path = config['LOG_PATH']
    logger = setup_logging(log_path)
    logger.info(f"🚀 Iniciando experimento: {experiment_name}")
    tracker = ExperimentTracker(config)  # Ajusta el constructor si es necesario
    return tracker, logger

def prepare_environment(logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Usando dispositivo: {device}")

    dataset = prepare_dataset(config['DATA_PATH'], config['DATASET'])
    logger.info(f"✅ Dataset cargado exitosamente. Total de muestras: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"🔄 DataLoader creado con batch_size={config['BATCH_SIZE']}")

    sample = dataset[0][0]
    if isinstance(sample, torch.Tensor):
        input_dim = sample.shape[0]
    elif isinstance(sample, np.ndarray):
        input_dim = sample.shape[0]
    else:
        raise TypeError("El tipo de datos de la muestra no es soportado.")

    return device, dataloader, input_dim

def configure_early_stopping(cfg):
    return EarlyStopping(
        patience=cfg.get('EARLY_STOPPING_PATIENCE', 10),
        min_delta=cfg.get('EARLY_STOPPING_MIN_DELTA', 0.001)
    )

def opt_learning_rate(lr_key, trial):
    lr_range = config['RANGES'][lr_key]
    low = lr_range['low']
    high = lr_range['high']
    if low > high:
        raise ValueError(f"El valor 'low' debe ser menor o igual a 'high' (low={low}, high={high}).")
    return trial.suggest_float(lr_key, low, high, log=True)

def objective(trial):
    global models
    try:
        # Ajustar hiperparámetros
        config['LEARNING_RATE_GENERATOR'] = opt_learning_rate('LEARNING_RATE_GENERATOR', trial)
        config['LEARNING_RATE_DISCRIMINATOR'] = opt_learning_rate('LEARNING_RATE_DISCRIMINATOR', trial)

        config['BETA1'] = trial.suggest_float(
            'BETA1',
            config['RANGES']['BETA1']['low'],
            config['RANGES']['BETA1']['high'],
            step=0.01
        )
        config['BETA2'] = trial.suggest_float(
            'BETA2',
            config['RANGES']['BETA2']['low'],
            config['RANGES']['BETA2']['high'],
            step=0.01
        )

        config['LATENT_DIM'] = trial.suggest_categorical('LATENT_DIM', config['RANGES']['LATENT_DIM'])
        config['BATCH_SIZE'] = trial.suggest_categorical('BATCH_SIZE', config['RANGES']['BATCH_SIZE'])
        config['EPOCHS'] = trial.suggest_int(
            'EPOCHS',
            config['RANGES']['EPOCHS']['low'],
            config['RANGES']['EPOCHS']['high']
        )

        device, dataloader, input_dim = prepare_environment(logger)
        models = initialize_models(config, input_dim, device, logger)
        early_stopping = configure_early_stopping(config)
        local_tracker = ExperimentTracker(config)

        train(models, dataloader, device, config, early_stopping, local_tracker)
        return local_tracker.get_final_loss()
    except Exception as e:
        logger.error(f"Error en la función objetivo de Optuna: {e}")
        return float('inf')

def optimize_experiment():
    global models, logger
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study.optimize(
        partial(objective),
        n_trials=50
    )

    if len(study.best_trials) > 0:
        logger.info(f"Mejores hiperparámetros: {study.best_params}")
    else:
        logger.error("No se completaron trials exitosos.")
        return

    # Actualiza la configuración
    config.update(study.best_params)

    if models is not None:
        save_models(models, config['MODEL_PATH'])
    else:
        logger.warning("No se encontró un modelo entrenado para guardar.")

    logger.info("📂 Resultados guardados exitosamente.")

def main():
    check_config_keys(REQUIRED_KEYS, config)
    
    global experiment_tracker
    experiment_tracker, _ = initialize_experiment("final_experiment")
    
    optimize_experiment()

if __name__ == "__main__":
    main()