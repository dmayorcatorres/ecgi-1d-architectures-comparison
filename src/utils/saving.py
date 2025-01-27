# utils/saving.py

import torch
import os
import json

def save_models(generator_state, discriminator_state, exp_dir):
    """
    Guarda los estados del Generador y Discriminador.

    Args:
        generator_state (dict): Estado del Generador.
        discriminator_state (dict): Estado del Discriminador.
        exp_dir (str): Directorio de experimentos.

    Returns:
        tuple: (ruta_generador, ruta_discriminador)
    """
    gen_path = os.path.join(exp_dir, "generator_best.pt")
    disc_path = os.path.join(exp_dir, "discriminator_best.pt")
    torch.save(generator_state, gen_path)  # No llamar a .state_dict()
    torch.save(discriminator_state, disc_path)
    return gen_path, disc_path

def save_best_trial(study, best_loss, best_g_losses, best_d_losses, exp_dir):
    """
    Guarda los mejores hiperparámetros y métricas en un archivo JSON.

    Args:
        study (optuna.study.Study): Estudio de Optuna.
        best_loss (float): Mejor pérdida obtenida.
        best_g_losses (list): Pérdidas del Generador.
        best_d_losses (list): Pérdidas del Discriminador.
        exp_dir (str): Directorio de experimentos.

    Returns:
        str: Ruta al archivo JSON guardado.
    """
    best_trial = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_loss': best_loss,
        'G_losses': best_g_losses,
        'D_losses': best_d_losses
    }

    best_trial_path = os.path.join(exp_dir, 'best_trial.json')
    with open(best_trial_path, 'w') as f:
        json.dump(best_trial, f, indent=4)
    
    return best_trial_path
