
import optuna
import torch
import datetime
import json
import logging
from src.utils.logger import handle_optimization_exception
from src.data.data_loader import load_data, create_dataset_and_loader
from src.models.simple_cgan import ConditionalGenerator, ConditionalDiscriminator
from src.training.trainer import trainCGAN
import src.utils.saving as saving
import src.utils.plotting as plotting
import os


# Función objetivo para Optuna
def objective(trial, config, exp_dir, device, logger):
    global BEST_LOSS, BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES

    try:
        logger.info(f"🔄 Iniciando Trial {trial.number}")
        mat_file = os.path.abspath(os.path.join(config['GENERAL']['DATA_PATH'], f"{config['GENERAL']['DATASET']}_data.mat"))
        data, tags_np = load_data(mat_file, logger)
        # Sugerir hiperparámetros
        latent_dim = trial.suggest_categorical('LATENT_DIM', config['RANGES']['LATENT_DIM'])
        batch_size = trial.suggest_categorical('BATCH_SIZE', config['RANGES']['BATCH_SIZE'])
        dropout = trial.suggest_categorical('DROPOUT', config['RANGES']['DROPOUT'])
        logger.info(f"🔄 Hiperparámetros sugeridos: LATENT_DIM={latent_dim}, BATCH_SIZE={batch_size}, DROPOUT={dropout}")

        # Crear DataLoader y obtener dimensión de entrada
        dataloader, input_dim = create_dataset_and_loader(data, tags_np, device, batch_size, logger)
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


def finalize_optimization(
    study,
    BEST_G_STATE,
    BEST_D_STATE,
    BEST_G_LOSSES,
    BEST_D_LOSSES,
    BEST_LOSS,
    exp_dir,
    logger
):
    """
    Refactor para manejar las tareas finales de optimización: 
    guardar los mejores modelos, registrar pérdidas, 
    guardar los mejores hiperparámetros y graficar la importancia.
    """
    if BEST_G_STATE is not None and BEST_D_STATE is not None:
        gen_path, disc_path = saving.save_models(BEST_G_STATE, BEST_D_STATE, exp_dir)
        logger.info(f"Modelos guardados: {gen_path}, {disc_path}")

        # Graficar pérdidas
        plot_path = plotting.plot_losses(BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
        logger.info(f"Gráfica de pérdidas guardada: {plot_path}")
    else:
        logger.info("No se guardó el modelo, no hubo mejoras o error en entrenamiento.")

    # Guardar los mejores hiperparámetros y métricas
    best_trial_path = saving.save_best_trial(study, BEST_LOSS, BEST_G_LOSSES, BEST_D_LOSSES, exp_dir)
    logger.info(f"Archivo con mejores hiperparámetros y métricas guardado: {best_trial_path}")

    # Visualizar historia de optimización e importancia de parámetros
    plot_history_path = plotting.plot_optimization_history(study, exp_dir)
    logger.info(f"Historia de optimización guardada: {plot_history_path}")

    plot_importances_path = plotting.plot_param_importances(study, exp_dir)
    logger.info(f"Importancia de hiperparámetros guardada: {plot_importances_path}")

    logger.info("Optimización finalizada.")