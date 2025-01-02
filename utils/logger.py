import logging
import os
from datetime import datetime

def setup_logging(experiment_path):
    """
    Configures the logging system for the experiment.

    Args:
        experiment_path (str): Full path to the experiment folder.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create logs directory within the experiment path
    log_dir = os.path.join(experiment_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure log file
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()
