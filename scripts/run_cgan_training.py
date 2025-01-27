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
import scipy.io as sio
from torch.utils.data import TensorDataset
import numpy as np
import logging

from src.configs import load_config
from src.utils.logger import setup_logging
from src.models.simple_cgan import ConditionalGenerator, ConditionalDiscriminator
import src.utils.saving as saving
import src.utils.plotting as plotting
from src.training.trainer import trainCGAN


# ----------------------------------------------------------
# VARIABLES GLOBALES para almacenar el mejor modelo
# ----------------------------------------------------------
BEST_LOSS = float('inf')
BEST_G_STATE = None
BEST_D_STATE = None
BEST_G_LOSSES = []
BEST_D_LOSSES = []

def load_data(config, logger):
    mat_file = os.path.abspath(os.path.join(config['GENERAL']['DATA_PATH'], f"{config['GENERAL']['DATASET']}_data.mat"))
    if not os.path.exists(mat_file):
        logger.error(f"El archivo de datos no existe: {mat_file}")
        raise FileNotFoundError(f"No se encontró el archivo de datos: {mat_file}")
    try:
        data = sio.loadmat(mat_file)
        logger.info(f"🔄 Datos cargados desde {mat_file}")
    except Exception as e:
        logger.error(f"Error al cargar el archivo .mat: {e}")
        raise
    raw_tags = data['tags'].squeeze(axis=1)
    tags_list = [item[0, 0] for item in raw_tags]
    tags_np = np.array(tags_list, dtype=np.float32) - 1
    return data, tags_np

# Función objetivo para Optuna
def objective(trial, config, exp_dir, device, logger):
    global BEST_LOSS, BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES

    try:
        logger.info(f"🔄 Iniciando Trial {trial.number}")
        data, tags_np = load_data(config, logger)

        # Procesar etiquetas
        tags = torch.from_numpy(tags_np).to(device)

        # Crear tensor de EGM
        egm = torch.tensor(data['egm'], dtype=torch.float32).to(device)
        egm_dataset = TensorDataset(egm, tags)

        # Sugerir hiperparámetros
        latent_dim = trial.suggest_categorical('LATENT_DIM', config['RANGES']['LATENT_DIM'])
        batch_size = trial.suggest_categorical('BATCH_SIZE', config['RANGES']['BATCH_SIZE'])
        dropout = trial.suggest_categorical('DROPOUT', config['RANGES']['DROPOUT'])

        input_dim = egm.shape[1]
        logger.info(f"🔄 Hiperparámetros sugeridos: LATENT_DIM={latent_dim}, BATCH_SIZE={batch_size}, DROPOUT={dropout}")

        # Crear DataLoader
        dataloader = torch.utils.data.DataLoader(
            egm_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        logger.info(f"🔄 DataLoader creado con batch_size={batch_size}.")

        # Crear modelos
        generator = ConditionalGenerator(
            latent_dim=latent_dim,
            num_classes=8,
            label_emb_dim=1,
            output_dim=input_dim,
            dropout_rate=dropout
        ).to(device)

        discriminator = ConditionalDiscriminator(
            input_dim=input_dim,
            num_classes=8,
            label_emb_dim=1,
            dropout_rate=dropout
        ).to(device)

        logger.info("🔄 Modelos Generador y Discriminador creados.")

        # Entrenar cGAN
        G_losses, D_losses = trainCGAN(generator, discriminator, dataloader, device, config, trial)
        combined_loss = 0.5 * G_losses[-1] + 0.5 * D_losses[-1]
        logger.info(f"[Trial {trial.number}] combined_loss={combined_loss:.4f}")

        # Reportar la pérdida a Optuna
        trial.report(combined_loss, step=1)

        # Decidir si se realiza pruning
        if trial.should_prune():
            logger.info(f"🔪 Trial {trial.number} podado.")
            raise optuna.exceptions.TrialPruned()

        # Actualizar el mejor modelo si es necesario
        if combined_loss < BEST_LOSS:
            BEST_LOSS = combined_loss
            BEST_G_STATE = generator.state_dict()
            BEST_D_STATE = discriminator.state_dict()
            BEST_G_LOSSES = G_losses.copy()
            BEST_D_LOSSES = D_losses.copy()
            logger.info(f"[Trial {trial.number}] => NUEVO MEJOR MODELO guardado! combined_loss={combined_loss:.4f}")

        return combined_loss

    except optuna.exceptions.TrialPruned:
        logger.info(f"🔪 Trial {trial.number} podado.")
        raise
    except Exception as e:
        logger.exception(f"Error en el trial {trial.number}: {e}")
        raise

if __name__ == "__main__":
    try:
        # 1. Directorio del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Nombre dinámico para la carpeta (por ejemplo, con fecha/hora)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # 9. Guardar el mejor modelo en .pt
        if BEST_G_STATE is not None and BEST_D_STATE is not None:
            gen_path, disc_path = saving.save_models(BEST_G_STATE, BEST_D_STATE, exp_dir)
            logger.info(f"Modelos guardados: {gen_path}, {disc_path}")

            # Graficar pérdidas
            plot_path = plotting.plot_losses(BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
            logger.info(f"Gráfica de pérdidas guardada: {plot_path}")
        else:
            logger.info("No se guardó el modelo, no hubo mejoras o error en entrenamiento.")

        # 10. Guardar los mejores hiperparámetros y métricas en un archivo JSON
        best_trial_path = saving.save_best_trial(study, BEST_LOSS, BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
        logger.info(f"Archivo con mejores hiperparámetros y métricas guardado: {best_trial_path}")

        # 11. Visualizar la historia de optimización y la importancia de los hiperparámetros
        plot_history_path = plotting.plot_optimization_history(study, exp_dir)
        logger.info(f"Historia de optimización guardada: {plot_history_path}")

        plot_importances_path = plotting.plot_param_importances(study, exp_dir)
        logger.info(f"Importancia de hiperparámetros guardada: {plot_importances_path}")

        logger.info("Optimización finalizada.")

    except Exception as e:
        # Manejar excepciones y registrar
        if 'logger' in locals():
            logger.exception("Ocurrió un error durante la optimización.")
        else:
            print(f"Error sin logger: {e}")
