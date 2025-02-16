import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
import logging
import torch

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ECGiDataset:
    FS: int = 1000  # Frecuencia de muestreo en Hz
    WAVELET: str = 'morl'  # Tipo de wavelet para CWT
    SCALES: np.ndarray = np.arange(1, 65)  # Escalas para CWT

    def __init__(self,
                 mat_file_path: str,
                 device: Optional[torch.device] = None) -> None:
        """
        Args:
            mat_file_path: Ruta al archivo .mat.
            device: Dispositivo PyTorch (cpu o cuda) donde cargar los tensores.
                    Si None, se detecta automáticamente ('cuda' si disponible, de lo contrario 'cpu').
        """
        self.mat_file_path = mat_file_path
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.data: Optional[dict] = None

        # Atributos donde se almacenan las señales y estructuras (como listas o arrays numpy)
        self.ecg: Optional[np.ndarray] = None
        self.emg: Optional[np.ndarray] = None

        # Estructuras de geometría (nodos, caras, etc.)
        self.heart_node: Optional[np.ndarray] = None
        self.heart_face: Optional[np.ndarray] = None
        self.torso_node: Optional[np.ndarray] = None
        self.torso_face: Optional[np.ndarray] = None

        # Otros campos
        self.tag: Optional[np.ndarray] = None
        self.lfm: Optional[np.ndarray] = None
        self.lam: Optional[np.ndarray] = None
        self.necg: Optional[int] = None
        self.nemg: Optional[int] = None

        # Cargamos los datos desde el .mat
        self.load_data()
    # -----------------------------------------------------
    # MÉTODOS PRINCIPALES DE CARGA DE DATOS
    # -----------------------------------------------------
    def load_data(self) -> None:
        """Carga datos del archivo .mat y extrae señales y estructuras."""
        try:
            self.data = sio.loadmat(self.mat_file_path,
                                    struct_as_record=False,
                                    squeeze_me=True)
            logger.info(f"Data loaded successfully from {self.mat_file_path}")
        except FileNotFoundError:
            logger.error(f"File {self.mat_file_path} not found.")
            return
        except Exception as e:
            logger.error(f"Error loading .mat file: {e}")
            return

        self.extract_structures(['heart', 'torso'])
        self.extract_signals(['ecg', 'emg'])
        self.extract_other_fields(['tag', 'lfm', 'lam', 'necg', 'nemg'])

    def extract_signals(self, signals: List[str]) -> None:
        """Extrae las señales especificadas (p.ej., 'ecg', 'emg') de self.data."""
        for signal_name in signals:
            signal_data = self.data.get(signal_name)
            if signal_data is None:
                logger.warning(f"Signal '{signal_name}' not found in the data.")
                continue

            cleaned_signals = self._clean_and_extract_signals(signal_data, signal_name)
            setattr(self, signal_name, cleaned_signals)
            logger.info(f"Loaded {len(cleaned_signals)} {signal_name.upper()} signals.")

    def _clean_and_extract_signals(self, signal_data: Union[np.ndarray, list], signal_name: str) -> np.ndarray:
        """
        Limpia y extrae señales desde cell arrays de MATLAB o arrays de NumPy.
        Cada elemento resultante suele ser un array de forma (n_canales, n_muestras).

        Retorna un np.ndarray de dtype=object, donde cada elemento es una de las señales.
        """
        if isinstance(signal_data, np.ndarray) and signal_data.dtype == object:
            signals_list = [s for s in signal_data.tolist() if s.size > 0]
            signals = np.array(signals_list, dtype=object).squeeze()
        else:
            if signal_data.size > 0:
                signals = np.array([signal_data], dtype=object).squeeze()
            else:
                signals = np.array([], dtype=object)
        return signals

    def extract_structures(self, structures: List[str]) -> None:
        """Extrae las estructuras especificadas (p.ej., 'heart', 'torso') y sus campos."""
        for structure in structures:
            struct_data = self.data.get(structure)
            if struct_data is None:
                logger.warning(f"Structure '{structure}' not found in the data.")
                continue

            self._extract_structure_fields(struct_data, structure)

    def _extract_structure_fields(self, struct_data, structure: str) -> None:
        """Extrae los campos 'node' y 'face' de una estructura."""
        for field in ['node', 'face']:
            field_data = getattr(struct_data, field, None)
            if field_data is not None:
                setattr(self, f"{structure}_{field}", np.array(field_data))
                logger.info(
                    f"{structure}.{field} loaded with shape: "
                    f"{self.get_shape(getattr(self, f'{structure}_{field}'))}"
                )
            else:
                logger.warning(f"{structure}.{field} not found in the data.")

    def extract_other_fields(self, fields: List[str]) -> None:
        """Extrae campos misceláneos de los datos."""
        for field in fields:
            field_data = self.data.get(field)
            if field_data is None:
                logger.warning(f"Field '{field}' not found in the data.")
                continue

            self._process_field(field, field_data)

    def _process_field(self, field: str, field_data: Union[np.ndarray, int]) -> None:
        """Procesa y almacena un campo según su tipo."""
        if field in ['necg', 'nemg']:
            setattr(self, field, int(field_data))
            logger.info(f"'{field}' loaded with value: {getattr(self, field)}")
        elif field == 'tag':
            tag_data = np.array(field_data).flatten()
            setattr(self, field, tag_data)
            logger.info(f"'{field}' loaded with shape: {tag_data.shape}")
        else:
            setattr(self, field, np.array(field_data))
            logger.info(
                f"'{field}' loaded with shape: {self.get_shape(getattr(self, field))}"
            )
    # -----------------------------------------------------
    # GETTERS PARA OBTENER TENSORES DE ECG Y EMG
    # -----------------------------------------------------
    def get_ecg_tensor(self) -> Optional[torch.Tensor]:
        if self.ecg is None or len(self.ecg) == 0:
            logger.warning("No ECG data available.")
            return None

        shapes = [arr.shape for arr in self.ecg]
        if len(set(shapes)) != 1:
            logger.warning("ECG signals have inconsistent shapes.")
            return None

        ecg_list_numeric = [arr.astype(np.float32) for arr in self.ecg]
        np_ecg_tensor = np.stack(ecg_list_numeric, axis=0)
        ecg_tensor = torch.tensor(np_ecg_tensor, dtype=torch.float32, device=self.device)
        logger.info(f"ECG tensor created with shape: {tuple(ecg_tensor.shape)} on {self.device}")
        return ecg_tensor

    def get_emg_tensor(self) -> Optional[torch.Tensor]:
        if self.emg is None or len(self.emg) == 0:
            logger.warning("No EMG data available.")
            return None

        shapes = [arr.shape for arr in self.emg]
        if len(set(shapes)) != 1:
            logger.warning("EMG signals have inconsistent shapes.")
            return None

        emg_list_numeric = [arr.astype(np.float32) for arr in self.emg]
        np_emg_tensor = np.stack(emg_list_numeric, axis=0)
        emg_tensor = torch.tensor(np_emg_tensor, dtype=torch.float32, device=self.device)
        logger.info(f"EMG tensor created with shape: {tuple(emg_tensor.shape)} on {self.device}")
        return emg_tensor


    # -----------------------------------------------------
    # UTILITARIO Y MÉTODO DE IMPRESIÓN
    # -----------------------------------------------------
    @staticmethod
    def get_shape(array: Optional[np.ndarray]) -> Optional[Tuple]:
        """Retorna la forma del array o None si no existe."""
        return array.shape if array is not None else None

    def print_summary(self) -> None:
        """Imprime un resumen de la información cargada."""
        print("\n--- Data Summary ---")
        print(f"ECG signals shape: {self.get_shape(self.ecg)}")
        print(f"EMG signals shape: {self.get_shape(self.emg)}")
        print(f"Heart node shape: {self.get_shape(self.heart_node)}")
        print(f"Heart face shape: {self.get_shape(self.heart_face)}")
        print(f"Torso node shape: {self.get_shape(self.torso_node)}")
        print(f"Torso face shape: {self.get_shape(self.torso_face)}")
        print(f"Tag shape: {self.get_shape(self.tag)}")
        print(f"LFM shape: {self.get_shape(self.lfm)}")
        print(f"LAM shape: {self.get_shape(self.lam)}")
        print(f"nECG: {self.necg}")
        print(f"nEMG: {self.nemg}")
        print(f"Device for PyTorch tensors: {self.device}")
        print("--------------------\n")