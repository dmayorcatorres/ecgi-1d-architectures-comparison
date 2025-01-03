import optuna
import torch
import os
import time
from configs import load_config
from models import SimpleGenerator, SimpleDiscriminator
from data import data_loader
import scipy.io as sio
from torch.utils.data import TensorDataset
from utils.logger import setup_logging
from torch.cuda.amp import autocast, GradScaler  # Correct import for autocast
from training.trainer import train

# Inicializar el logger
logger = setup_logging()  # Asegúrate de que esta función devuelva un objeto logger válido
config = load_config()

# Función objetivo para Optuna
def objective(trial):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"🔄 Load device={device}")

    data_path = config['GENERAL']['DATA_PATH']
    dataset_name = config['GENERAL']['DATASET']
    logger.info(f"🔄 Load data_path={data_path}, dataset_name={dataset_name}")

    mat_file = os.path.join(data_path, f"{dataset_name}_data.mat")
    data = sio.loadmat(mat_file)
    #ecg = torch.tensor(data['ecg'], dtype=torch.float32).to(device)
    egm = torch.tensor(data['egm'], dtype=torch.float32).to(device)
    #tags = torch.tensor(data['tags'], dtype=torch.float32).to(device)
    egm_dataset = TensorDataset(egm)

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

    generator = SimpleGenerator(latent_dim, input_dim, dropout).to(device)
    discriminator = SimpleDiscriminator(input_dim, latent_dim, dropout).to(device)

    G_losses, D_losses  = train(generator, discriminator, dataloader, device, config, trial)
    combined_loss = 0.5 * G_losses[-1] + 0.5 * D_losses[-1]
    return combined_loss


# Crear un estudio y optimizar
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")  # Cambiar a "minimize" si optimizas pérdida
    study.optimize(objective, n_trials=50)
    # Mostrar el mejor conjunto de hiperparámetros
    print("Best hyperparameters:", study.best_params)
    print("Best score:", study.best_value)
