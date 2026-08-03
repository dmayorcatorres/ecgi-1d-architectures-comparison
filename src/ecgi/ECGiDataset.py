import scipy.io as sio
import numpy as np
import torch
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Configuración centralizada de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

@dataclass(frozen=True)
class DatasetConfig:
    wavelet: str = 'morl'
    scales: np.ndarray = field(default_factory=lambda: np.arange(1, 65))
    fs: int = 1000
    # Ahora solo son obligatorios estos campos:
    required_structures: Tuple[str, ...] = ('heart', 'torso')
    required_signals: Tuple[str, ...] = ('ecg', 'emg')
    numeric_fields: Tuple[str, ...] = ('necg', 'nemg')
    array_fields: Tuple[str, ...] = ('tag',)  # Solo 'tags' es obligatorio en arrays

class ECGiDataset:
    def __init__(self,
                 mat_file_path: str,
                 config: DatasetConfig = DatasetConfig(),
                 device: Optional[torch.device] = None):
        """
        Args:
            mat_file_path: Ruta al archivo .mat.
            config: Configuración del dataset (inmutable).
            device: Dispositivo para tensores PyTorch.
        """
        self.mat_file_path = mat_file_path
        self.config = config
        self.device = device or self._auto_detect_device()

        self.signals: Dict[str, np.ndarray] = {}
        self.structures: Dict[str, Dict[str, np.ndarray]] = {}
        self.metadata: Dict[str, Union[int, np.ndarray]] = {}

        self._load_data()

    def _auto_detect_device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_data(self) -> None:
        try:
            raw_data = self._load_mat_file()
            self._validate_data_structure(raw_data)
            self._process_raw_data(raw_data)
            logger.info(f"Data loaded successfully from {self.mat_file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_mat_file(self) -> Dict[str, Any]:
        try:
            return sio.loadmat(self.mat_file_path, struct_as_record=False, squeeze_me=True)
        except FileNotFoundError:
            logger.error(f"File not found: {self.mat_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading .mat file: {str(e)}")
            raise

    def _validate_data_structure(self, raw_data: Dict[str, Any]) -> None:
        """
        Valida que los campos marcados como 'obligatorios' existan en el archivo .mat.
        Si algún campo requerido no está presente, lanza una excepción.
        """
        required_fields = (
            list(self.config.required_structures)
            + list(self.config.required_signals)
            + list(self.config.numeric_fields)
            + list(self.config.array_fields)
        )

        missing = [field for field in required_fields if field not in raw_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    def _process_raw_data(self, raw_data: Dict[str, Any]) -> None:
        self._extract_structures(raw_data)
        self._extract_signals(raw_data)
        self._extract_metadata(raw_data)

    def _extract_structures(self, raw_data: Dict[str, Any]) -> None:
        for structure in self.config.required_structures:
            struct_data = raw_data.get(structure)
            if struct_data is None:
                continue
            self.structures[structure] = {
                'node': self._get_numpy_array(getattr(struct_data, 'node', None)),
                'face': self._get_numpy_array(getattr(struct_data, 'face', None))
            }

    def _extract_signals(self, raw_data: Dict[str, Any]) -> None:
        for signal in self.config.required_signals:
            signal_data = raw_data.get(signal)
            if signal_data is None:
                continue
            cleaned_signals = self._clean_signal_data(signal_data)
            self.signals[signal] = cleaned_signals

    def _clean_signal_data(self, signal_data: Union[np.ndarray, list]) -> np.ndarray:
        if isinstance(signal_data, np.ndarray) and signal_data.dtype == object:
            return np.array([s.astype(np.float32) for s in signal_data.squeeze()])
        return np.array(signal_data, dtype=np.float32)

    def _extract_metadata(self, raw_data: Dict[str, Any]) -> None:
        for field in self.config.numeric_fields:
            self.metadata[field] = int(raw_data.get(field, 0))

        # 'tags' es el único campo de array obligatorio; los demás array-fields ya son opcionales.
        # Si quieres procesar otros arrays opcionales, podrías hacer algo como:
        # optional_array_fields = ['lfm', 'lam']  # u otros nombres
        # for field in optional_array_fields:
        #     if field in raw_data:
        #         self.metadata[field] = self._get_numpy_array(raw_data.get(field))

        for field in self.config.array_fields:
            self.metadata[field] = self._get_numpy_array(raw_data.get(field))

    def _get_numpy_array(self, data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        array = np.array(data)
        return array if array.size > 0 else None

    def _validate_signal_shapes(self, signals: np.ndarray) -> bool:
        if signals.size == 0:
            logger.warning("No signals available")
            return False
        if len(set(s.shape for s in signals)) > 1:
            logger.warning("Inconsistent signal shapes detected")
            return False
        return True

    def _get_tensor(self, signal_type: str) -> Optional[torch.Tensor]:
        if signal_type not in self.signals:
            logger.warning(f"No {signal_type.upper()} data available")
            return None
        signals = self.signals[signal_type]
        if not self._validate_signal_shapes(signals):
            return None
        return torch.tensor(
            np.stack(signals),
            dtype=torch.float32,
            device=self.device
        )

    @property
    def ecg_tensor(self) -> Optional[torch.Tensor]:
        return self._get_tensor('ecg')

    @property
    def emg_tensor(self) -> Optional[torch.Tensor]:
        return self._get_tensor('emg')

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            'signals': {k: np.array(v).shape for k, v in self.signals.items()},
            'structures': {
                k: {f: (s.shape if s is not None else None)
                    for f, s in v.items()}
                for k, v in self.structures.items()
            },
            'metadata': self.metadata,
            'device': str(self.device)
        }

    # Propiedades de acceso a las mallas
    @property
    def heart_node(self) -> Optional[np.ndarray]:
        return self.structures.get('heart', {}).get('node', None)

    @property
    def heart_face(self) -> Optional[np.ndarray]:
        return self.structures.get('heart', {}).get('face', None)

    @property
    def torso_node(self) -> Optional[np.ndarray]:
        return self.structures.get('torso', {}).get('node', None)

    @property
    def torso_face(self) -> Optional[np.ndarray]:
        return self.structures.get('torso', {}).get('face', None)

    def __repr__(self) -> str:
        return (f"ECGiDataset(mat_file_path='{self.mat_file_path}', "
                f"signals={list(self.signals.keys())}, "
                f"structures={list(self.structures.keys())})")

    def print_summary(self) -> None:
        """
        Imprime un resumen amigable de las dimensiones de señales,
        estructuras y metadatos.
        """
        summary_dict = {
            'signals': {k: np.array(v).shape for k, v in self.signals.items()},
            'structures': {
                k: {f: (s.shape if s is not None else None)
                    for f, s in v.items()}
                for k, v in self.structures.items()
            },
            'metadata': self.metadata,
            'device': str(self.device)
        }
        logger.info("=== ECGiDataset Summary ===")

        # Imprimir shapes de señales
        if 'signals' in summary_dict:
            for sig_key, shape_ in summary_dict['signals'].items():
                logger.info(f"Signal '{sig_key}': shape = {shape_}")
        # Imprimir shapes de estructuras
        if 'structures' in summary_dict:
            for struct_name, fields in summary_dict['structures'].items():
                logger.info(f"Structure '{struct_name}':")
                for f_name, f_shape in fields.items():
                    logger.info(f"  - {f_name}: {f_shape}")
        # Imprimir metadatos
        if 'metadata' in summary_dict:
            logger.info(f"Metadata: {summary_dict['metadata']}")
        # Dispositivo
        if 'device' in summary_dict:
            logger.info(f"Device: {summary_dict['device']}")