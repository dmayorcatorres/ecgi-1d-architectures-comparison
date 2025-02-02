import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pywt
from scipy import signal


class ECGiDataset:
    # === Parámetros de Configuración Iniciales ===
    FS = 1000  # Frecuencia de muestreo en Hz
    WAVELET = 'morl'  # Tipo de wavelet para CWT
    SCALES = np.arange(1, 65)  # Escalas para CWT
    LOWCUT = 0.5  # Frecuencia de corte inferior para filtro (Hz)
    HIGHCUT = 200  # Frecuencia de corte superior para filtro (Hz)
    FILTER_ORDER = 5  # Orden del filtro Butterworth

    # Constantes para configuración del modelo de Keras
    NUM_FILTERS = 64
    KERNEL_SIZE = 5
    DENSE_UNITS = 512
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.5

    def __init__(self, mat_file_path: str):
        """
        Inicializa la clase ECGiDataset cargando los datos desde el archivo .mat y construyendo el modelo de extracción de características.

        Parámetros:
        - mat_file_path: str, ruta al archivo .mat que contiene los datos.
        """
        self.mat_file_path = mat_file_path
        self.data = None
        self.heart_node = None
        self.heart_face = None
        self.torso_node = None
        self.torso_face = None
        self.ecg = []
        self.emg = []
        self.tag = None
        self.lfm = None
        self.lam = None
        self.necg = None
        self.nemg = None

        # Modelo de extracción de características
        self.feature_extractor = None

        self.load_data()

    def load_data(self):
        """
        Carga los datos desde el archivo .mat y asigna los atributos correspondientes.
        """
        try:
            # Cargar el archivo .mat con opciones adecuadas
            self.data = sio.loadmat(self.mat_file_path, struct_as_record=False, squeeze_me=True)
            print(f"Datos cargados exitosamente desde {self.mat_file_path}")
        except FileNotFoundError:
            print(f"Error: Archivo {self.mat_file_path} no encontrado.")
            return
        except Exception as e:
            print(f"Error al cargar el archivo .mat: {e}")
            return

        # Extraer y asignar estructuras 'heart' y 'torso'
        self.extract_structures(['heart', 'torso'])

        # Extraer y asignar señales 'ecg' y 'emg'
        self.extract_signals(['ecg', 'emg'])

        # Extraer otros campos
        self.extract_other_fields(['tag', 'lfm', 'lam', 'necg', 'nemg'])

    def extract_structures(self, structures: List[str]):
        """
        Extrae las estructuras especificadas ('heart', 'torso') y sus campos 'node' y 'face'.

        Parámetros:
        - structures: List[str], nombres de las estructuras a extraer.
        """
        for structure in structures:
            struct_data = self.data.get(structure, None)
            if struct_data is not None:
                try:
                    nodes = getattr(struct_data, 'node', None)
                    faces = getattr(struct_data, 'face', None)

                    if nodes is not None:
                        setattr(self, f"{structure}_node", np.array(nodes))
                        print(f"{structure}.node cargado con forma: {self.get_shape(getattr(self, f'{structure}_node'))}")
                    else:
                        print(f"Advertencia: {structure}.node no encontrado en los datos.")

                    if faces is not None:
                        setattr(self, f"{structure}_face", np.array(faces))
                        print(f"{structure}.face cargado con forma: {self.get_shape(getattr(self, f'{structure}_face'))}")
                    else:
                        print(f"Advertencia: {structure}.face no encontrado en los datos.")
                except AttributeError as e:
                    print(f"Error al acceder a los campos de {structure}: {e}")
            else:
                print(f"Advertencia: Estructura '{structure}' no encontrada en los datos.")

    def extract_signals(self, signals: List[str]):
        """
        Extrae las señales especificadas ('ecg', 'emg') que están almacenadas como celdas en MATLAB.

        Parámetros:
        - signals: List[str], nombres de las señales a extraer.
        """
        for signal_name in signals:
            signal_data = self.data.get(signal_name, None)
            if signal_data is not None:
                if isinstance(signal_data, np.ndarray) and signal_data.dtype == object:
                    # Convertir a lista si es un array de objetos (celdas en MATLAB)
                    extracted_signals = signal_data.tolist()
                    # Filtrar señales vacías
                    extracted_signals = [s for s in extracted_signals if s.size > 0]
                    setattr(self, signal_name, extracted_signals)
                    print(f"Se cargaron {len(extracted_signals)} señales {signal_name.upper()} (filtradas {len(signal_data) - len(extracted_signals)} vacías).")
                else:
                    if signal_data.size > 0:
                        setattr(self, signal_name, [signal_data])
                        print(f"Se cargó una única señal {signal_name.upper()}.")
                    else:
                        setattr(self, signal_name, [])
                        print(f"Advertencia: Señal {signal_name.upper()} está vacía.")
            else:
                print(f"Advertencia: Señal '{signal_name}' no encontrada en los datos.")

    def extract_other_fields(self, fields: List[str]):
        """
        Extrae otros campos especificados.

        Parámetros:
        - fields: List[str], nombres de los campos a extraer.
        """
        for field in fields:
            field_data = self.data.get(field, None)
            if field_data is not None:
                if field in ['necg', 'nemg']:
                    setattr(self, field, int(field_data))
                    print(f"'{field}' cargado con valor: {getattr(self, field)}")
                elif field == 'tag':
                    setattr(self, field, np.array(field_data).flatten())
                    print(f"'{field}' cargado con forma: {self.tag.shape}")
                else:
                    setattr(self, field, np.array(field_data))
                    print(f"'{field}' cargado con forma: {self.get_shape(getattr(self, field))}")
            else:
                print(f"Advertencia: Campo '{field}' no encontrado en los datos.")

    def get_shape(self, array: Optional[np.ndarray]) -> Optional[Tuple]:
        """
        Retorna la forma del array o 'None' si el array es None.

        Parámetros:
        - array: np.ndarray o None

        Retorna:
        - Tuple o None
        """
        return array.shape if array is not None else None

    def print_summary(self):
        """
        Imprime un resumen de todas las estructuras y señales cargadas.
        """
        print("\n--- Resumen de los Datos Cargados ---")
        print(f"heart_node shape: {self.get_shape(self.heart_node)}")
        print(f"heart_face shape: {self.get_shape(self.heart_face)}")
        print(f"torso_node shape: {self.get_shape(self.torso_node)}")
        print(f"torso_face shape: {self.get_shape(self.torso_face)}")
        print(f"Número de señales ECG: {len(self.ecg)}")
        print(f"Número de señales EMG: {len(self.emg)}")
        print(f"tags shape: {self.tag.shape if self.tag is not None else 'None'}")
        print(f"lfm shape: {self.get_shape(self.lfm)}")
        print(f"lam shape: {self.get_shape(self.lam)}")
        print(f"necg: {self.necg}")
        print(f"nemg: {self.nemg}")
        print("-------------------------------------\n")

    def print_signal_info(self, signal_type: str, index: int):
        """
        Imprime información detallada sobre una señal específica.

        Parámetros:
        - signal_type: str, 'ecg' o 'emg'.
        - index: int, índice de la señal a imprimir (0-based).
        """
        if signal_type.lower() not in ['ecg', 'emg']:
            print("Error: 'signal_type' debe ser 'ecg' o 'emg'.")
            return

        signals = self.ecg if signal_type.lower() == 'ecg' else self.emg

        if index < 0 or index >= len(signals):
            print(f"Error: Índice fuera de rango. Hay {len(signals)} señales {signal_type.upper()}.")
            return

        signal = signals[index]
        print(f"\n--- Información de la Señal {signal_type.upper()} {index + 1} ---")
        print(f"Tipo: {type(signal)}")
        print(f"Forma: {signal.shape}")
        print(f"Datos (primeros 5 elementos): {signal.flatten()[:5]}")
        print("------------------------------------------\n")

    def plot_ecg_signal(self, index: int, electrode_index: Optional[int] = None):
        """
        Visualiza una señal ECG específica. Si se proporciona 'electrode_index', dibuja solo ese electrodo.
        Si no, dibuja todas las señales de los electrodos.

        Parámetros:
        - index: int, índice de la señal ECG a visualizar (0-based).
        - electrode_index: Optional[int], índice del electrodo a visualizar (0-based). Si es None, dibuja todos.
        """
        if index < 0 or index >= len(self.ecg):
            print(f"Error: Índice fuera de rango. Hay {len(self.ecg)} señales ECG.")
            return

        signal = self.ecg[index]

        if electrode_index is not None:
            if electrode_index < 0 or electrode_index >= signal.shape[0]:
                print(f"Error: Índice de electrodo fuera de rango. Hay {signal.shape[0]} electrodos.")
                return

            electrode_signal = signal[electrode_index, :]
            plt.figure(figsize=(12, 4))
            plt.plot(electrode_signal, label=f'ECG Signal {index + 1} - Electrode {electrode_index + 1}')
            plt.title(f'Señal ECG {index + 1} - Electrode {electrode_index + 1}')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Dibujar todas las señales de los electrodos en subplots
            num_electrodes = signal.shape[0]
            num_cols = 4
            num_rows = int(np.ceil(num_electrodes / num_cols))
            plt.figure(figsize=(20, num_rows * 3))
            for i in range(num_electrodes):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.plot(signal[i, :], color='blue')
                plt.title(f'Electrode {i + 1}')
                plt.xlabel('Tiempo (s)')
                plt.ylabel('Amplitud')
                plt.tight_layout()
            plt.suptitle(f'Señal ECG {index + 1} - Todos los Electrodos', fontsize=16, y=1.02)
            plt.show()

    def plot_emg_signal(self, index: int, channel_index: Optional[int] = None):
        """
        Visualiza una señal EMG específica. Si se proporciona 'channel_index', dibuja solo ese canal.
        Si no, dibuja todas las señales de los canales en subplots.

        Parámetros:
        - index: int, índice de la señal EMG a visualizar (0-based).
        - channel_index: Optional[int], índice del canal a visualizar (0-based). Si es None, dibuja todos.
        """
        if index < 0 or index >= len(self.emg):
            print(f"Error: Índice fuera de rango. Hay {len(self.emg)} señales EMG.")
            return

        signal = self.emg[index]

        if channel_index is not None:
            if channel_index < 0 or channel_index >= signal.shape[0]:
                print(f"Error: Índice de canal fuera de rango. Hay {signal.shape[0]} canales.")
                return

            channel_signal = signal[channel_index, :]
            plt.figure(figsize=(12, 4))
            plt.plot(channel_signal, label=f'EMG Signal {index + 1} - Channel {channel_index + 1}')
            plt.title(f'Señal EMG {index + 1} - Channel {channel_index + 1}')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # Dibujar todas las señales de los canales en subplots
            num_channels = signal.shape[0]
            num_cols = 4
            num_rows = int(np.ceil(num_channels / num_cols))
            plt.figure(figsize=(20, num_rows * 3))
            for i in range(num_channels):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.plot(signal[i, :], color='green')
                plt.title(f'Channel {i + 1}')
                plt.xlabel('Tiempo (s)')
                plt.ylabel('Amplitud')
                plt.tight_layout()
            plt.suptitle(f'Señal EMG {index + 1} - Todos los Canales', fontsize=16, y=1.02)
            plt.show()

    def get_first_ecg_signal(self) -> Optional[np.ndarray]:
        """
        Retorna la primera señal ECG si existe.

        Retorna:
        - np.ndarray o None
        """
        return self.ecg[0] if self.ecg else None

    def get_first_emg_signal(self) -> Optional[np.ndarray]:
        """
        Retorna la primera señal EMG si existe.

        Retorna:
        - np.ndarray o None
        """
        return self.emg[0] if self.emg else None

    def preprocess_input(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocesa la señal ECG o EMG para que tenga la forma adecuada para el modelo.

        Parámetros:
        - signal: np.ndarray, forma (163, 200) para ECG o (C, T) para EMG

        Retorna:
        - signal_processed: np.ndarray, forma (1, 200, 163, 1) para ECG o (1, T, C, 1) para EMG
        """
        # Detectar si es ECG o EMG basado en la forma
        if signal.ndim != 2:
            raise ValueError("La señal debe ser una matriz 2D (electrodos/canales x tiempo).")

        if signal.shape[0] == 163 and signal.shape[1] == 200:
            # ECG Signal
            processed_signal = np.expand_dims(signal, axis=0)  # (1, 163, 200)
            processed_signal = np.transpose(processed_signal, (0, 2, 1))  # (1, 200, 163)
            processed_signal = np.expand_dims(processed_signal, axis=-1)  # (1, 200, 163, 1)
            return processed_signal
        else:
            # EMG Signal
            processed_signal = np.expand_dims(signal, axis=0)  # (1, C, T)
            processed_signal = np.transpose(processed_signal, (0, 2, 1))  # (1, T, C)
            processed_signal = np.expand_dims(processed_signal, axis=-1)  # (1, T, C, 1)
            return processed_signal


    def compute_scalogram(self, ecg_index: int, electrode_index: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calcula el escalograma (scalogram) de una señal ECG específica y de un electrodo determinado.

        Parámetros:
        - ecg_index: int, índice de la señal ECG a procesar (0-based).
        - electrode_index: int, índice del electrodo a analizar (0-based).

        Retorna:
        - Tuple containing:
            - scalogram: np.ndarray, matriz de escalograma (frecuencias x tiempo)
            - frequencies: np.ndarray, array de frecuencias correspondientes a las escalas
            - times: np.ndarray, array de tiempos correspondientes a los datos
        - None, si ocurre un error.
        """
        # Validar índice de ECG
        if ecg_index < 0 or ecg_index >= len(self.ecg):
            print(f"Error: Índice de ECG fuera de rango. Hay {len(self.ecg)} señales ECG.")
            return None

        ecg_signal = self.ecg[ecg_index]

        # Validar índice de electrodo
        if electrode_index < 0 or electrode_index >= ecg_signal.shape[0]:
            print(f"Error: Índice de electrodo fuera de rango. Hay {ecg_signal.shape[0]} electrodos.")
            return None

        # Extraer la señal del electrodo específico
        electrode_signal = ecg_signal[electrode_index, :]  # Forma: (200,)

        # Aplicar filtro de banda pasante (opcional, si no está filtrada previamente)
        filtered_signal = self.bandpass_filter(electrode_signal, self.LOWCUT, self.HIGHCUT, self.FILTER_ORDER)

        # Calcular CWT
        coefs, frequencies = pywt.cwt(filtered_signal, self.SCALES, self.WAVELET, sampling_period=1.0 / self.FS)

        # Calcular el escalograma (magnitud de los coeficientes)
        scalogram = np.abs(coefs)

        # Definir el vector de tiempos
        times = np.linspace(0, len(filtered_signal) / self.FS, num=len(filtered_signal))

        return scalogram, frequencies, times

    def bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
        """
        Aplica un filtro de banda pasante Butterworth a la señal.

        Parámetros:
        - data: np.ndarray, señal a filtrar.
        - lowcut: float, frecuencia de corte inferior (Hz).
        - highcut: float, frecuencia de corte superior (Hz).
        - order: int, orden del filtro.

        Retorna:
        - y: np.ndarray, señal filtrada.
        """
        nyq = 0.5 * self.FS
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data)
        return y

    def compute_and_concatenate_scalograms(self, ecg_index: int) -> Optional[np.ndarray]:
        """
        Calcula y concatena los escalogramas de todos los electrodos para una señal ECG específica.

        Parámetros:
        - ecg_index: int, índice de la señal ECG a procesar (0-based).

        Retorna:
        - concatenated_scalograms: np.ndarray, matriz concatenada de escalogramas de todos los electrodos.
        - None, si ocurre un error.
        """
        if ecg_index < 0 or ecg_index >= len(self.ecg):
            print(f"Error: Índice de ECG fuera de rango. Hay {len(self.ecg)} señales ECG.")
            return None

        ecg_signal = self.ecg[ecg_index]
        num_electrodes = ecg_signal.shape[0]

        scalograms = []

        for electrode in range(num_electrodes):
            result = self.compute_scalogram(ecg_index, electrode)
            if result is None:
                print(f"Saltando electrodo {electrode} debido a fallo en la computación del escalograma.")
                continue
            scalogram, frequencies, times = result
            scalograms.append(scalogram)

        if not scalograms:
            print("Error: No se pudieron computar escalogramas para ningún electrodo.")
            return None

        # Apilar los escalogramas a lo largo de una nueva dimensión (canales)
        concatenated_scalograms = np.stack(scalograms, axis=-1)  # (frecuencias, tiempo, num_electrodes)

        print(f"Concatenated Scalograms shape: {concatenated_scalograms.shape}")

        return concatenated_scalograms

    def compute_and_plot_scalogram(self, ecg_index: int, electrode_index: int):
        """
        Calcula y visualiza el escalograma de un electrodo específico dentro de una señal ECG.

        Parámetros:
        - ecg_index: int, índice de la señal ECG a procesar (0-based).
        - electrode_index: int, índice del electrodo a analizar (0-based).
        """
        result = self.compute_scalogram(ecg_index, electrode_index)
        if result is None:
            print("No se pudo calcular el escalograma.")
            return

        scalogram, frequencies, times = result

        plt.figure(figsize=(10, 6))
        plt.imshow(scalogram, extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                   cmap='viridis', aspect='auto')
        plt.title(f'Escalograma - ECG {ecg_index + 1} - Electrode {electrode_index + 1}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        plt.colorbar(label='Magnitud')
        plt.tight_layout()
        plt.show()