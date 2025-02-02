import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset

def load_data(mat_file, logger):
    
    if not os.path.exists(mat_file):
        logger.error(f"El archivo de datos no existe: {mat_file}")
        raise FileNotFoundError(f"No se encontró el archivo de datos: {mat_file}")
    try:
        data = sio.loadmat(mat_file)
        logger.info(f"🔄 Datos cargados desde {mat_file}")
    except Exception as e:
        logger.error(f"Error al cargar el archivo .mat: {e}")
        raise
    raw_tags = data['tags'].squeeze(axis=1)
    tags_list = [item[0, 0] for item in raw_tags]
    tags_np = np.array(tags_list, dtype=np.float32) - 1
    return data, tags_np

def create_dataset_and_loader(data, tags_np, device, batch_size, logger):
    """
    Crea el TensorDataset y DataLoader a partir de los datos EGM y las etiquetas.
    Retorna el dataloader y la dimensión de entrada.
    """
    tags = torch.from_numpy(tags_np).to(device)
    egm = torch.tensor(data['egm'], dtype=torch.float32).to(device)
    
    egm_dataset = TensorDataset(egm, tags)
    # imprimir tamaño de egm y tags como logger
    logger.info(f"🔄 Dataset creado con egm: {egm.shape} tag. {tags.shape} registros.")
    # Contar la cantidad de clases en tag
    n_classes = len(torch.unique(tags))
    logger.info(f"🔄 Número de clases: {n_classes}")
    # contar por clase
    for i in range(n_classes):
        count = (tags == i).sum().item()
        logger.info(f"🔄 Clase {i}: {count} registros.")



    dataloader = torch.utils.data.DataLoader(
        egm_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    input_dim = egm.shape[1]
    return dataloader, input_dim