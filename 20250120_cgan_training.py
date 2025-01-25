import optuna
import torch
import os
import time
from configs import load_config
from models.simple_cgan import ConditionalGenerator, ConditionalDiscriminator
from data import data_loader
import scipy.io as sio
from torch.utils.data import TensorDataset
from utils.logger import setup_logging
from torch.cuda.amp import autocast, GradScaler  # Correct import for autocast
from training.trainer import trainCGAN
import numpy as np
import datetime
# Graficar pérdidas
import matplotlib.pyplot as plt
import json

# ----------------------------------------------------------
# VARIABLES GLOBALES para almacenar el mejor modelo
# ----------------------------------------------------------
BEST_LOSS = float('inf')
BEST_G_STATE = None
BEST_D_STATE = None
BEST_G_LOSSES = []
BEST_D_LOSSES = []

# Función objetivo para Optuna
def objective(trial):
    global BEST_LOSS, BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔄 Load device={device}")

    data_path = config['GENERAL']['DATA_PATH']
    dataset_name = config['GENERAL']['DATASET']
    logger.info(f"🔄 Load data_path={data_path}, dataset_name={dataset_name}")

    mat_file = os.path.join(data_path, f"{dataset_name}_data.mat")
    data = sio.loadmat(mat_file)
    print(type(data['tags']))
    print(data['tags'].dtype)
    print(data['tags'].shape)
    #ecg = torch.tensor(data['ecg'], dtype=torch.float32).to(device)
    egm = torch.tensor(data['egm'], dtype=torch.float32).to(device)

    raw_tags = data['tags']  # dtype=object, shape: (N, 1), con cada elemento = array([[8]], dtype=uint8)
    # 1) Elimina la dimensión sobrante (pasa de (N,1) a (N,))
    raw_tags = raw_tags.squeeze(axis=1)

    # 2) Extrae el valor interno de cada objeto array([[x]])
    tags_list = []
    for item in raw_tags:
        # item es algo como array([[8]], dtype=uint8)
        value = item[0, 0]  # extrae 8
        tags_list.append(value)

    # 3) Convierte a un array NumPy (float o int, según requieras)
    tags_np = np.array(tags_list, dtype=np.float32)  # por ejemplo float32
    tags_np = tags_np - 1

    # 4) Crea el tensor de PyTorch
    tags = torch.from_numpy(tags_np).to(device)
    egm_dataset = TensorDataset(egm, tags)


    latent_dim = trial.suggest_categorical('LATENT_DIM', config['RANGES']['LATENT_DIM'])
    batch_size = trial.suggest_categorical('BATCH_SIZE', config['RANGES']['BATCH_SIZE'])
    dropout = trial.suggest_categorical('DROPOUT', config['RANGES']['DROPOUT'])

    input_dim = egm.shape[1]
    logger.info(f"🔄 Load latent_dim={latent_dim}, input_dim={input_dim}")

    dataloader = torch.utils.data.DataLoader(
        egm_dataset,
        batch_size= batch_size,
        num_workers=0
    )
    num_classes = 8
    label_emb_dim = 1
    label_shape = (1,)

    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        label_emb_dim=label_emb_dim,
        output_dim=input_dim,
        dropout_rate=dropout
    ).to(device)

    discriminator = ConditionalDiscriminator(
        input_dim=input_dim,
        num_classes=num_classes,
        label_emb_dim=label_emb_dim,
        dropout_rate=dropout
    ).to(device)

    G_losses, D_losses  = trainCGAN(generator, discriminator, dataloader, device, config, trial)
    combined_loss = 0.5 * G_losses[-1] + 0.5 * D_losses[-1]

    logger.info(f"[Trial {trial.number}] combined_loss={combined_loss:.4f}")

    # Si es el mejor, guardamos sus datos
    if combined_loss < BEST_LOSS:
        BEST_LOSSES = combined_loss
        BEST_G_STATE = generator.state_dict()
        BEST_D_STATE = discriminator.state_dict()
        BEST_G_LOSSES = G_losses
        BEST_D_LOSSES = D_losses
        logger.info(f"[Trial {trial.number}] => NUEVO MEJOR MODELO guardado! combined_loss={combined_loss:.4f}")


    return combined_loss


# Crear un estudio y optimizar
if __name__ == "__main__":

    # Inicializar el logger
    logger = setup_logging()  # Asegúrate de que esta función devuelva un objeto logger válido
    config = load_config()
    logger.info(f"🔄 Load config={config}")

    # 1. Directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Nombre dinámico para la carpeta (por ejemplo, con fecha/hora)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(script_dir, "experiments/",f"experiment_{now_str}")

    # 3. Crear carpeta
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"🔄 Creando carpeta de experimento {exp_dir}"
                )
    study = optuna.create_study(direction="minimize")  # Cambiar a "minimize" si optimizas pérdida
    study.optimize(objective, n_trials=100)

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best score: {study.best_value}")

   # global BEST_LOSS, BEST_G_STATE, BEST_D_STATE, BEST_G_LOSSES, BEST_D_LOSSES
    logger.info(f"BEST_LOSS: {BEST_LOSS}")

    # Guardar el mejor modelo en .pt
    if BEST_G_STATE is not None:
        gen_path = os.path.join(exp_dir, "generator_best.pt")
        disc_path = os.path.join(exp_dir, "discriminator_best.pt")
        torch.save(BEST_G_STATE, gen_path)
        torch.save(BEST_D_STATE, disc_path)
        logger.info(f"Modelos guardados: {gen_path}, {disc_path}")

        plt.figure()
        plt.plot(BEST_G_LOSSES, label="G_loss")
        plt.plot(BEST_D_LOSSES, label="D_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = os.path.join(exp_dir, "loss_plot.png")
        plt.savefig(plot_path)
        logger.info(f"Gráfica de pérdidas guardada: {plot_path}")
    else:
        logger.info("No se guardó el modelo, no hubo mejoras o error en entrenamiento.")

            # 9. Guardar los mejores hiperparámetros y métricas en un archivo JSON
    best_trial = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_loss': BEST_LOSS,
        'G_losses': BEST_G_LOSSES,
        'D_losses': BEST_D_LOSSES
    }

    best_trial_path = os.path.join(exp_dir, 'best_trial.json')
    with open(best_trial_path, 'w') as f:
        json.dump(best_trial, f, indent=4)

    logger.info(f"Archivo con mejores hiperparámetros y métricas guardado: {best_trial_path}")

    logger.info("Optimización finalizada.")

