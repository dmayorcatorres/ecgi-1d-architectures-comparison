import sys
import os
import torch
import mlflow
import numpy as np
from datetime import datetime
import random
import optuna
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# IMPORT CONFIGURATION FROM A FILE OR DEFINE IT DIRECTLY
from configs import config  # Ensure config.py exists and is imported correctly

from data.data_loader import prepare_dataset
from models.factory import create_gan_model
from training.trainer import train, initialize_models  # Import train and initialize_models
from utils.logger import setup_logging
from utils.experiment_tracker import ExperimentTracker
from utils.early_stopping import EarlyStopping
from utils.saver import save_loss_plot, save_models  # Import save_loss_plot and save_models

def check_config_keys(required_keys):
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

def initialize_experiment(experiment_name="final_experiment"):
    """Initialize the experiment and configure paths and logging."""
    experiment_tracker = ExperimentTracker(experiment_name)
    config['MODEL_PATH'] = experiment_tracker.get_model_path()
    config['PLOT_PATH'] = experiment_tracker.get_plot_path()
    config['LOG_PATH'] = experiment_tracker.get_log_path()

    logger = setup_logging(experiment_tracker.get_log_path())
    logger.info("🚀 Starting experiment...")

    return experiment_tracker, logger

def prepare_environment(logger):
    """Set up the training environment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")
    
    # Ensure `config['DATA_PATH']` and `config['DATASET']` are correctly defined in `config.py`
    dataset = prepare_dataset(config['DATA_PATH'], config['DATASET'])  
    logger.info(f"✅ Dataset loaded successfully. Total samples: {len(dataset)}")
    
    return device, dataset, dataset[0][0].shape[0]

def configure_early_stopping():
    """Configure early stopping."""
    return EarlyStopping(
        patience=config.get('EARLY_STOPPING_PATIENCE', 10),
        min_delta=config.get('EARLY_STOPPING_MIN_DELTA', 0.001)
    )

def objective(trial, config):
    """Objective function for Optuna."""
    try:
        # Define hyperparameters to optimize
        config['LEARNING_RATE_GENERATOR'] = trial.suggest_float('LEARNING_RATE_GENERATOR', 1e-5, 1e-2, log=True)
        config['LEARNING_RATE_DISCRIMINATOR'] = trial.suggest_float('LEARNING_RATE_DISCRIMINATOR', 1e-5, 1e-2, log=True)
        config['LATENT_DIM'] = trial.suggest_categorical('LATENT_DIM', [32, 64, 128])
        config['BETA1'] = trial.suggest_float('BETA1', 0.5, 0.9)
        config['BETA2'] = trial.suggest_float('BETA2', 0.9, 0.999)
        config['BATCH_SIZE'] = trial.suggest_categorical('BATCH_SIZE', [16, 32, 64])
        config['EPOCHS'] = trial.suggest_int('EPOCHS', 100, 300)

        # Initialize the experiment, prepare the environment and models
        experiment_tracker, logger = initialize_experiment()
        device, dataset, num_cols = prepare_environment(logger)
        generator, discriminator = initialize_models(config, num_cols, device, logger)
        
        # Configure early stopping
        early_stopping = configure_early_stopping()
        
        # Execute training
        G_losses, D_losses = train(generator, discriminator, dataset, config, device, logger, experiment_tracker, early_stopping)
        
        # Log performance metrics
        validation_loss = experiment_tracker.get_metrics().get('final_loss', None)
        
        # Log the best parameters in the experiment
        experiment_tracker.log_config(config)
        
        # Save the plots generated during training
        save_loss_plot(G_losses, D_losses, config['PLOT_PATH'])

        # Save the models
        save_models(generator, discriminator, config['MODEL_PATH'])

        if validation_loss is None:
            return float('inf')  # Return a large float value to indicate a failed trial

        return validation_loss  # This is the metric that Optuna will use to minimize
    except Exception as e:
        print(f"Error in objective function: {e}")  # Debugging
        return float('inf')  # If there is an error, Optuna will take it as a bad result

def main():
    """Main function to run the GAN experiment with Optuna."""
    required_keys = ['LATENT_DIM', 'LR', 'BETA1', 'BETA2', 'BATCH_SIZE', 'EPOCHS', 'DATA_PATH', 'DATASET']
    check_config_keys(required_keys)

    # Use a common folder for all experiments
    experiment_name = "final_experiment"  # Change if you need a specific name
    experiment_tracker, logger = initialize_experiment(experiment_name)

    # Create the Optuna study
    study = optuna.create_study(direction='minimize')  # 'minimize' if we are minimizing an error metric
    
    # Here, we pass `config` as an argument to the objective function
    study.optimize(lambda trial: objective(trial, config), n_trials=10)  # Perform 10 trials (adjust the number of trials as needed)

    # Check if there are successful trials
    if len(study.trials) > 0:
        print("Best hyperparameters: ", study.best_params)
    else:
        print("No successful trials completed.")

    # Run the experiment with the best parameters found
    config.update(study.best_params)  # Update the configuration with the best parameters

    # Continue with the execution and save results in the same folder
    # Save the best results in the metrics file at the end
    metrics = experiment_tracker.get_metrics()
    best_loss = metrics.get('final_loss', None)
    if best_loss is not None:
        experiment_tracker.log_summary({
            'best_loss': best_loss,
            'best_params': study.best_params,
        })
    
    # Create a summary with Pandas (Excel)
    results = pd.DataFrame({
        'Config': [config],
        'Final Loss': [best_loss],
        'Best Params': [study.best_params]
    })
    results.to_excel(os.path.join(experiment_tracker.get_experiment_path(), 'summary_results.xlsx'), index=False)

    logger.info("🎯 Experiment completed successfully.")
    logger.info(f"📂 Results in: {experiment_tracker.get_experiment_path()}")

if __name__ == "__main__":
    main()