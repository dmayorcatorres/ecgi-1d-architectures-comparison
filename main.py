import sys
import os
import torch
import mlflow
import numpy as np
from datetime import datetime
from configs import config
from data.data_loader import prepare_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from training.trainer import train
from utils.logger import setup_logging
from utils.experiment_tracker import ExperimentTracker
from utils.early_stopping import EarlyStopping

def initialize_experiment():
    # Ensure the root directory is in sys.path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    # Generate experiment name based on date and time
    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize Experiment Tracker
    experiment_tracker = ExperimentTracker(experiment_name)
    config['MODEL_PATH'] = experiment_tracker.get_model_path()
    config['PLOT_PATH'] = experiment_tracker.get_plot_path()
    config['LOG_PATH'] = experiment_tracker.get_log_path()
    
    # Setup Logging
    logger = setup_logging(experiment_tracker.get_log_path())
    logger.info("🚀 Starting experiment...")
    
    return experiment_tracker, logger

def prepare_environment(logger):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")
    
    # Prepare dataset
    dataset = prepare_dataset(config['DATA_PATH'], config['DATASET'])
    logger.info(f"✅ Dataset loaded successfully. Total samples: {len(dataset)}")
    
    # Get dataset dimensions
    num_rows = len(dataset)
    num_cols = dataset[0][0].shape[0] if len(dataset) > 0 else 0
    logger.info(f"📊 Rows: {num_rows}, Columns: {num_cols}")
    
    return device, dataset, num_cols

def initialize_models(num_cols, device, logger):
    # Initialize models
    generator = Generator(config['LATENT_DIM'], num_cols).to(device)
    discriminator = Discriminator(num_cols, config['LATENT_DIM']).to(device)
    logger.info(f"📝 Generator and Discriminator initialized.")
    
    return generator, discriminator

def configure_early_stopping():
    # Configure Early Stopping
    return EarlyStopping(
        patience=config.get('EARLY_STOPPING_PATIENCE', 10),
        min_delta=config.get('EARLY_STOPPING_MIN_DELTA', 0.001)
    )

def save_models(generator, discriminator, experiment_tracker, logger):
    # Save final models
    generator_path = os.path.join(config['MODEL_PATH'], "generator_final.pt")
    discriminator_path = os.path.join(config['MODEL_PATH'], "discriminator_final.pt")
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    
    logger.info(f"✅ Final Generator model saved at: {generator_path}")
    logger.info(f"✅ Final Discriminator model saved at: {discriminator_path}")

def main():
    experiment_tracker, logger = initialize_experiment()
    device, dataset, num_cols = prepare_environment(logger)
    generator, discriminator = initialize_models(num_cols, device, logger)
    
    experiment_tracker.log_experiment_info(logger)
    experiment_tracker.log_config(config)
    
    early_stopping = configure_early_stopping()
    
    # Start MLflow Tracking
    mlflow.set_experiment("DeepLearning_Experiments")

    with mlflow.start_run():
        mlflow.log_params(config)
        
        try:
            # Start training
            train(generator, discriminator, dataset, config, device, logger, experiment_tracker, early_stopping)
            
            # Create an input example for MLflow
            input_example = torch.randn(1, config['LATENT_DIM'], device=device).cpu().float().numpy()
            
            # Log models in MLflow
            mlflow.pytorch.log_model(
                generator,
                artifact_path="generator",
                registered_model_name="GAN_Generator",
                signature=None
            )

            mlflow.pytorch.log_model(
                discriminator,
                artifact_path="discriminator",
                registered_model_name="GAN_Discriminator",
                signature=None
            )
  
            logger.info("✅ Models successfully saved in MLflow.")
        
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            raise e
        
        finally:
            save_models(generator, discriminator, experiment_tracker, logger)
    
    logger.info("🎯 Experiment successfully completed.")
    logger.info(f"📂 Results saved at: {experiment_tracker.get_experiment_path()}")

if __name__ == "__main__":
    main()