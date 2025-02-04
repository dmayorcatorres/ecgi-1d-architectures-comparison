import optuna
import torch
import datetime
import os
import logging
from src.utils.logger import handle_optimization_exception
from src.data.data_loader import load_data, create_dataset_and_loader
from src.training.trainerManager import TrainerManager
import src.utils.saving as saving
import src.utils.plotting as plotting


# ----------------------------------------------------------
# VARIABLES GLOBALES para almacenar el mejor modelo
# ----------------------------------------------------------
BEST_LOSS = float('inf')
BEST_G_STATE = None
BEST_D_STATE = None
BEST_G_LOSSES = []
BEST_D_LOSSES = []


def objective(trial, config, exp_dir, device, logger):
    """
    Función objetivo para la optimización con Optuna.
    
    - Carga los datos
    - Define y entrena un modelo (GAN o CGAN)
    - Evalúa el rendimiento y actualiza el mejor modelo
    
    :param trial: Instancia de Optuna para la optimización de hiperparámetros
    :param config: Configuración del experimento
    :param exp_dir: Directorio de salida del experimento
    :param device: Dispositivo (CPU o GPU)
    :param logger: Logger para registrar eventos
    :return: Pérdida combinada del modelo evaluado
    """
    global BEST_LOSS, BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES

    try:
        logger.info(f"🔄 Iniciando Trial {trial.number}")

        # Cargar datos
        mat_file = os.path.abspath(config['GENERAL']['DATA_PATH'])
        data, tags_np = load_data(mat_file, logger)

        # Crear DataLoader y obtener dimensión de entrada
        batch_size = trial.suggest_categorical('BATCH_SIZE', config['TRAINING']['BATCH_SIZE'])
        dataloader, input_dim = create_dataset_and_loader(data, tags_np, device, batch_size, logger)

        # Crear manejador de entrenamiento
        trainer_manager = TrainerManager(config, exp_dir, device, logger)

        # Seleccionar el modelo a entrenar en este trial
        model_type = trial.suggest_categorical("MODEL_TYPE", config['GENERAL']['MODEL_TYPE'])
        generator, discriminator, G_losses, D_losses = trainer_manager.train_model(trial, model_type, dataloader, input_dim)

        # Calcular la pérdida combinada
        combined_loss = 0.5 * G_losses[-1] + 0.5 * D_losses[-1]
        logger.info(f"[Trial {trial.number}] {model_type} Loss={combined_loss:.4f}")

        # Si se encuentra un mejor modelo, actualizar las variables globales
        if combined_loss < BEST_LOSS:
            BEST_LOSS = combined_loss
            BEST_G_STATE = generator.state_dict()
            BEST_D_STATE = discriminator.state_dict()
            BEST_G_LOSSES = G_losses.copy()
            BEST_D_LOSSES = D_losses.copy()
            logger.info(f"✅ Nuevo mejor modelo guardado con pérdida: {combined_loss:.4f}")

        return combined_loss

    except optuna.exceptions.TrialPruned:
        logger.info(f"🔪 Trial {trial.number} podado por bajo rendimiento.")
        raise
    except Exception as e:
        logger.exception(f"❌ Error en el trial {trial.number}: {e}")
        raise


def finalize_optimization(study, exp_dir, logger):
    """
    Guarda los mejores modelos, las pérdidas y la historia de optimización.

    :param study: Estudio de Optuna con los mejores hiperparámetros
    :param exp_dir: Directorio donde se guardarán los modelos y métricas
    :param logger: Logger para registrar eventos
    """
    global BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES, BEST_LOSS

    if BEST_G_STATE is not None and BEST_D_STATE is not None:
        # Guardar los mejores modelos
        gen_path, disc_path = saving.save_models(BEST_G_STATE, BEST_D_STATE, exp_dir)
        logger.info(f"📌 Modelos guardados: {gen_path}, {disc_path}")

        # Guardar y graficar las pérdidas
        plot_path = plotting.plot_losses(BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
        logger.info(f"📊 Gráfica de pérdidas guardada en: {plot_path}")

    else:
        logger.warning("⚠ No se guardó ningún modelo, no hubo mejoras.")

    # Guardar los mejores hiperparámetros y métricas
    best_trial_path = saving.save_best_trial(study, BEST_LOSS, BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
    logger.info(f"📄 Hiperparámetros y métricas guardados en: {best_trial_path}")

    # Visualizar historia de optimización e importancia de parámetros
    plot_history_path = plotting.plot_optimization_history(study, exp_dir)
    logger.info(f"📉 Historia de optimización guardada en: {plot_history_path}")

    plot_importances_path = plotting.plot_param_importances(study, exp_dir)
    logger.info(f"📊 Importancia de hiperparámetros guardada en: {plot_importances_path}")

    logger.info("✅ Optimización finalizada.")
