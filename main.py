import sys
import os
import torch
import mlflow
import numpy as np
from datetime import datetime

# Asegurar que el directorio raíz esté en sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs import config
from data.data_loader import prepare_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.trainer import train
from utils.logger import setup_logging
from utils.experiment_tracker import ExperimentTracker
from utils.early_stopping import EarlyStopping

# 📝 Generar automáticamente el nombre del experimento basado en la fecha y hora
experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 📂 Inicializar Tracker de Experimentos
experiment_tracker = ExperimentTracker(experiment_name)
config['MODEL_PATH'] = experiment_tracker.get_model_path()
config['PLOT_PATH'] = experiment_tracker.get_plot_path()
config['LOG_PATH'] = experiment_tracker.get_log_path()

# 📝 Configurar Logging
logger = setup_logging(experiment_tracker.get_log_path())
logger.info("🚀 Iniciando experimento...")

# 🖥️ Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️ Usando dispositivo: {device}")

# 📊 Preparar dataset
dataset = prepare_dataset(config['DATA_PATH'], config['DATASET'])
logger.info(f"✅ Dataset cargado correctamente. Total de muestras: {len(dataset)}")

# 📏 Obtener dimensiones del dataset
num_rows = len(dataset)
num_cols = dataset[0][0].shape[0] if len(dataset) > 0 else 0
logger.info(f"📊 Filas: {num_rows}, Columnas: {num_cols}")

# 🛠️ Inicializar modelos
generator = Generator(config['LATENT_DIM'], num_cols).to(device)
discriminator = Discriminator(num_cols, config['LATENT_DIM']).to(device)
logger.info(f"📝 Generador y Discriminador inicializados.")

# 📂 Registrar información del experimento
experiment_tracker.log_experiment_info(logger)
# Guardar configuración
experiment_tracker.log_config(config)


# 🛡️ Configurar Early Stopping
early_stopping = EarlyStopping(
    patience=config.get('EARLY_STOPPING_PATIENCE', 10),
    min_delta=config.get('EARLY_STOPPING_MIN_DELTA', 0.001)
)

# 🚀 Iniciar MLflow Tracking
mlflow.set_experiment("DeepLearning_Experiments")

with mlflow.start_run():
    # Registrar parámetros en MLflow
    mlflow.log_params(config)

    try:
        # 🔄 Iniciar entrenamiento
        train(generator, discriminator, dataset, config, device, logger, experiment_tracker, early_stopping)
        
        # ✅ Crear un ejemplo de entrada compatible con MLflow
        input_example = torch.randn(1, config['LATENT_DIM'], device=device).cpu().float().numpy()
        
        # 📝 Guardar modelos en MLflow
        mlflow.pytorch.log_model(
            generator,
            artifact_path="generator",
            input_example=input_example,
            registered_model_name="GAN_Generator"
        )
        
        mlflow.pytorch.log_model(
            discriminator,
            artifact_path="discriminator",
            input_example=input_example,
            registered_model_name="GAN_Discriminator"
        )
        
        logger.info("✅ Modelos guardados exitosamente en MLflow.")
    
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        raise e

    finally:
        # ✅ Guardar los modelos finales
        generator_path = os.path.join(config['MODEL_PATH'], "generator_final.pt")
        discriminator_path = os.path.join(config['MODEL_PATH'], "discriminator_final.pt")
        
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        
        logger.info(f"✅ Modelo final del Generador guardado en: {generator_path}")
        logger.info(f"✅ Modelo final del Discriminador guardado en: {discriminator_path}")

# ✅ Fin del experimento
logger.info("🎯 Experimento completado con éxito.")
logger.info(f"📂 Resultados guardados en: {experiment_tracker.get_experiment_path()}")
