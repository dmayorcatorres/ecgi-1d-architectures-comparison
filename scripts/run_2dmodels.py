import sys
import os
import math
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader

# --- Ajuste de PYTHONPATH para entornos locales ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ecgi.ECGiDataset import ECGiDataset, DatasetConfig
    from src.ecgi.ECGMetricsComparator import ECGMetricsComparator
    from src.ecgi.ECGiVisualizer import ECGiVisualizer
    from src.ecgi.SignalProcessorTorch import SignalProcessorTorch
    from src.models.cnn_models import RawCNNBase
    from src.ecgi.ResultsSaver import ResultsSaver
    from src.ecgi.TimeSeriesTo2DTransform import TimeSeriesTo2DTransform
except ModuleNotFoundError as e:
    logging.error("No se pudo encontrar el paquete 'src'. Verifica la estructura del proyecto y el PYTHONPATH.")
    raise e

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    # Ruta al archivo de datos (ajusta la ruta según tu estructura)
    data_path = os.path.join(os.path.dirname(__file__), "data", "raw", "auckland", "auckland_data.mat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar el dataset
    dataset = ECGiDataset(data_path, device=device)
    emg_tensor = dataset.emg_tensor.cpu()  # Tensor de EMG: (num_samples, input_channels, time_steps)
    ecg_tensor = dataset.ecg_tensor.cpu()  # Tensor de ECG: (num_samples, output_channels, time_steps)
    
    # Imprimir las dimensiones de los tensores
    print("Tamaño de emg_tensor:", emg_tensor.shape)
    print("Tamaño de ecg_tensor:", ecg_tensor.shape)
    
    # Crear un DataLoader sencillo (batch_size=1, por ejemplo)
    ds = TensorDataset(emg_tensor, ecg_tensor)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Determinar parámetros para el modelo según los tensores cargados
    time_steps = emg_tensor.shape[2]
    input_channels = emg_tensor.shape[1]
    output_channels = ecg_tensor.shape[1]
    
    # Instanciar el modelo (en este ejemplo se usa la variante "lite")
    model = RawCNNBase(variant="lite",
                       time_steps=time_steps,
                       input_channels=input_channels,
                       output_channels=output_channels)
    model.to(device)
    
    # Realizar un pase hacia adelante con un batch de ejemplo
    for batch in loader:
        emg_batch, ecg_batch = batch
        emg_batch = emg_batch.to(device)
        output = model(emg_batch)
        print("Salida del modelo (shape):", output.shape)
        break  # Solo se prueba un batch

if __name__ == "__main__":
    main()
