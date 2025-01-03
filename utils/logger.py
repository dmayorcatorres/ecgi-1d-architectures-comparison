import logging

def setup_logging():
    logger = logging.getLogger('GAN')
    logger.setLevel(logging.INFO)

    # Configuración del handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Formato de los logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
