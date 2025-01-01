import logging
import os
from datetime import datetime

def setup_logging(experiment_path):
    """
    Configura el sistema de logs para el experimento.

    Args:
        experiment_path (str): Ruta completa de la carpeta del experimento.

    Returns:
        logging.Logger: Logger configurado.
    """
    # Crear carpeta de logs dentro de la ruta del experimento
    log_dir = os.path.join(experiment_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar archivo de log
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(os.path.basename(experiment_path))
    logger.info(f"📝 Logs guardados en: {log_file}")
    return logger
