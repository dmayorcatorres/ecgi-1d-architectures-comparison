import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset

def load_data(data_path, dataset_name):
    """
    Carga los datos desde un archivo .mat.
    Args:
        data_path (str): Ruta al directorio de datos.
        dataset_name (str): Nombre del dataset.
    Returns:
        tuple: Datos cargados desde el archivo.
    """
    mat_file = os.path.join(data_path, f"{dataset_name}_data.mat")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"El archivo {mat_file} no existe.")
    
    data = sio.loadmat(mat_file)
    return data['ecg'], data['egm'], np.array(data['tags']), data['A']

def prepare_dataset(data_path, dataset_name):
    """
    Prepara un dataset PyTorch.
    Args:
        data_path (str): Ruta al directorio de datos.
        dataset_name (str): Nombre del dataset.
    Returns:
        TensorDataset: Dataset PyTorch listo para su uso.
    """
    _, y, _, _ = load_data(data_path, dataset_name)
    y = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(y)