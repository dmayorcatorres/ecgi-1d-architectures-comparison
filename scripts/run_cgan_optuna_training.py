# scripts/run_cgan_training.py

import sys
import os

# Añadir la raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import optuna
import torch
import datetime
import json
import logging

from src.configs import load_config
from src.utils.logger import setup_logging
import src.utils.saving as saving
import src.utils.plotting as plotting
from src.training.optimizer import objective, finalize_optimization, handle_optimization_exception



# ----------------------------------------------------------
# VARIABLES GLOBALES para almacenar el mejor modelo
# ----------------------------------------------------------
BEST_LOSS = float('inf')
BEST_G_STATE = None
BEST_D_STATE = None
BEST_G_LOSSES = []
BEST_D_LOSSES = []


if __name__ == "__main__":
    try:
        # 1. Directorio del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Nombre dinámico para la carpeta (por ejemplo, con fecha/hora)
        now_str = datetime.datetime.now().strftime("%Y%mdd_%H%M%S")
        exp_dir = os.path.join(script_dir, "experiments", f"experiment_{now_str}")

        # 3. Crear carpeta
        os.makedirs(exp_dir, exist_ok=True)

        # 4. Inicializar el logger
        logger = setup_logging(exp_dir)
        logger.info(f"🔄 Creando carpeta de experimento {exp_dir}")

        # 5. Cargar configuración
        config = load_config()
        logger.info(f"🔄 Load config={config}")

        # 6. Configurar el dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔄 Usando dispositivo: {device}")

        # 7. Crear estudio y optimizar usando Optuna con soporte para pruning
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(trial, config, exp_dir, device, logger), n_trials=100)

        # 8. Loguear los mejores resultados
        logger.info(f"Mejores hiperparámetros: {study.best_params}")
        logger.info(f"Mejor score: {study.best_value}")
        logger.info(f"BEST_LOSS: {BEST_LOSS}")

        # 9. Finalizar la optimización
        finalize_optimization(
            study,
            BEST_G_STATE,
            BEST_D_STATE,
            BEST_G_LOSSES,
            BEST_D_LOSSES,
            BEST_LOSS,
            exp_dir,
            logger
        )

    except Exception as e:
        handle_optimization_exception(e, logger)
