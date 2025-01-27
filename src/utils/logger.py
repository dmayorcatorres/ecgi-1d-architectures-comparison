# utils/logger.py

import logging
import os

def setup_logging(exp_dir=None):
    """
    Configura el logger para escribir en consola y en un archivo.

    Args:
        exp_dir (str, optional): Directorio donde se guardará el archivo de logs.

    Returns:
        logging.Logger: Instancia configurada del logger.
    """
    logger = logging.getLogger("GAN")
    logger.setLevel(logging.INFO)
    
    # Evitar agregar múltiples handlers
    if not logger.handlers:
        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
        
        # Handler de archivo si exp_dir está proporcionado
        if exp_dir is not None:
            log_file_path = os.path.join(exp_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
    
    return logger
