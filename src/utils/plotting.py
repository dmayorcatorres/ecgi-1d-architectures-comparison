# utils/plotting.py

import matplotlib.pyplot as plt
import os
import optuna.visualization as vis
import logging

def plot_losses(g_losses, d_losses, exp_dir):
    """
    Grafica las pérdidas del Generador y Discriminador y las guarda.

    Args:
        g_losses (list): Pérdidas del Generador.
        d_losses (list): Pérdidas del Discriminador.
        exp_dir (str): Directorio de experimentos.

    Returns:
        str: Ruta al archivo de la gráfica de pérdidas.
    """
    logger = logging.getLogger("GAN")
    logger.info("Generando gráfica de pérdidas.")
    plt.figure()
    plt.plot(g_losses, label="G_loss")
    plt.plot(d_losses, label="D_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(exp_dir, "loss_plot.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfica de pérdidas guardada en {plot_path}.")
    return plot_path

def plot_optimization_history(study, exp_dir):
    """
    Grafica la historia de optimización de Optuna y la guarda.

    Args:
        study (optuna.study.Study): Estudio de Optuna.
        exp_dir (str): Directorio de experimentos.

    Returns:
        str: Ruta al archivo de la gráfica de historia de optimización.
    """
    logger = logging.getLogger("GAN")
    logger.info("Generando gráfica de historia de optimización.")
    fig = vis.plot_optimization_history(study)
    fig_path = os.path.join(exp_dir, 'optimization_history.png')
    fig.write_image(fig_path)
    logger.info(f"Historia de optimización guardada en {fig_path}.")
    return fig_path

def plot_param_importances(study, exp_dir):
    """
    Grafica la importancia de los hiperparámetros y la guarda.

    Args:
        study (optuna.study.Study): Estudio de Optuna.
        exp_dir (str): Directorio de experimentos.

    Returns:
        str: Ruta al archivo de la gráfica de importancia de hiperparámetros.
    """
    logger = logging.getLogger("GAN")
    logger.info("Generando gráfica de importancia de hiperparámetros.")
    fig = vis.plot_param_importances(study)
    fig_path = os.path.join(exp_dir, 'param_importances.png')
    fig.write_image(fig_path)
    logger.info(f"Importancia de hiperparámetros guardada en {fig_path}.")
    return fig_path
