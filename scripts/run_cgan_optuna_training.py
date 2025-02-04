import sys
import os
import optuna
import torch
import datetime
import logging

# Añadir la raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importaciones del proyecto
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


def main():
    """
    Función principal para la optimización de modelos GAN/CGAN con Optuna.
    """
    try:
        # 1️⃣ Directorio del script actual y creación de directorio de experimentos
        script_dir = os.path.dirname(os.path.abspath(__file__))
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(script_dir, "experiments", f"experiment_{now_str}")
        os.makedirs(exp_dir, exist_ok=True)

        # 2️⃣ Inicializar el logger
        logger = setup_logging(exp_dir)
        logger.info(f"📂 Creando carpeta de experimento: {exp_dir}")

        # 3️⃣ Cargar configuración
        config = load_config()
        logger.info(f"⚙️ Configuración cargada: {config}")

        # 4️⃣ Configurar el dispositivo (CPU o GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Usando dispositivo: {device}")

        # 5️⃣ Crear estudio Optuna y optimizar con pruning
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(trial, config, exp_dir, device, logger), n_trials=100)

        # 6️⃣ Loguear los mejores resultados
        logger.info(f"🏆 Mejores hiperparámetros encontrados: {study.best_params}")
        logger.info(f"📉 Mejor score obtenido: {study.best_value}")
        logger.info(f"🔎 BEST_LOSS: {BEST_LOSS}")

        # 7️⃣ Finalizar la optimización y guardar modelos
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


# Ejecutar la función principal
if __name__ == "__main__":
    main()
