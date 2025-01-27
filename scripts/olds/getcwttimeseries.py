import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import pywt

# Parámetros
DATASET = "utah"  # Actualiza el nombre del conjunto de datos
freq = 1000  # Frecuencia de muestreo (Hz)
dt = 1 / freq
wavelet_name = 'morl'  # Wavelet continua compatible

# Obtener el directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas de datos y archivo
DATA_PATH = os.path.join(script_dir, 'data', 'utah')  # Ruta relativa a 'data/kitpvc'
DATA_FILENAME = 'utah_data.mat'  # Nombre del archivo de datos

# Directorios para guardar resultados
IMAGES_DIR = os.path.join(script_dir, 'images')
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

TENSOR_DIR = os.path.join(script_dir, 'tensor')
if not os.path.exists(TENSOR_DIR):
    os.makedirs(TENSOR_DIR)

SIGNALS_DIR = os.path.join(script_dir, 'signals')
if not os.path.exists(SIGNALS_DIR):
    os.makedirs(SIGNALS_DIR)

RESULTS_DIR = os.path.join(script_dir, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 1. Cargar datos desde el archivo .mat
def load_data_mat():
    full_file_path = os.path.join(DATA_PATH, DATA_FILENAME)
    print(f"Cargando datos desde: {full_file_path}")
    if not os.path.exists(full_file_path):
        raise FileNotFoundError(f"Error: El archivo '{DATA_FILENAME}' no se encontró en '{full_file_path}'. Por favor, verifica la ruta y asegúrate de que exista.")
    
    mat_data = sio.loadmat(full_file_path)
    print(f"Variables disponibles en el archivo: {list(mat_data.keys())}")
    A = mat_data['A'] if 'A' in mat_data else None  # Ajusta según tus datos
    X = mat_data['ecg']
    y = mat_data['egm']
    label = np.array(mat_data['tags']) if 'tags' in mat_data else None  # Ajusta según tus datos

    # Imprimir tamaños de las matrices cargadas
    print(f"Tamaño de X (ecg): {X.shape}")
    print(f"Tamaño de y (egm): {y.shape}")
    if label is not None:
        print(f"Tamaño de label: {label.shape}")
    if A is not None:
        print(f"Tamaño de A: {A.shape}")

    return X, y, label, A

# 2. Implementación del algoritmo de Pan-Tompkins
def pan_tompkins_detector(ecg_signal, fs):
    # Paso 1: Filtrado Bandapás (5-15 Hz)
    lowcut = 5.0
    highcut = 15.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    order = 1  # Orden del filtro

    b, a = signal.butter(order, [low, high], btype='band')
    ecg_band = signal.lfilter(b, a, ecg_signal)

    # Paso 2: Derivación
    ecg_diff = np.diff(ecg_band)
    ecg_diff = np.append(ecg_diff, 0)  # Alinear tamaños

    # Paso 3: Cuadrado
    ecg_squared = ecg_diff ** 2

    # Paso 4: Integración mediante Ventana Móvil
    window_size = int(0.150 * fs)  # 150 ms de ventana
    ecg_integrated = np.convolve(ecg_squared, np.ones(window_size)/window_size, mode='same')

    # Paso 5: Umbralización y Detección de Picos
    threshold = np.mean(ecg_integrated) * 1.5  # Ajusta el factor según sea necesario
    peaks_indices = np.where(ecg_integrated > threshold)[0]

    # Refina la detección de picos
    r_peaks = []
    refractory_period = int(0.200 * fs)  # 200 ms de período refractario
    last_peak = -np.inf

    for index in peaks_indices:
        if index - last_peak > refractory_period:
            # Buscar el pico máximo en una ventana alrededor del índice
            window_indices = range(max(0, index - int(0.050 * fs)), min(len(ecg_signal), index + int(0.050 * fs)))
            peak = np.argmax(ecg_signal[window_indices]) + window_indices.start
            r_peaks.append(peak)
            last_peak = peak

    return np.array(r_peaks)

# Función para calcular la CWT de una señal
def calcular_cwt(signal, scales, wavelet_name, fs):
    """
    Calcula los coeficientes de la CWT para una señal dada.

    Parámetros:
        signal: arreglo numpy de la señal a analizar
        scales: escalas para la CWT
        wavelet_name: nombre de la wavelet a utilizar
        fs: frecuencia de muestreo

    Retorna:
        coefficients: coeficientes CWT
        frequencies: frecuencias correspondientes a las escalas
    """
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet_name, sampling_period=1/fs)
    return coefficients, frequencies

# Función para visualizar y guardar el scalograma
def guardar_scalograma(coefficients, frequencies, t, filename, titulo='Scalograma CWT'):
    """
    Guarda el scalograma como una imagen.

    Parámetros:
        coefficients: coeficientes CWT
        frequencies: frecuencias correspondientes a las escalas
        t: vector de tiempo
        filename: ruta del archivo donde se guardará la imagen
        titulo: título del gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.contourf(t, frequencies, np.abs(coefficients), cmap='jet')
    plt.title(titulo)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.colorbar(label='Magnitud')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scalograma guardado en: {filename}")

if __name__ == "__main__":
    # Cargar datos
    X_np, y_np, label, A = load_data_mat()
    
    # Definir el directorio para guardar las señales y las imágenes
    if not os.path.exists(SIGNALS_DIR):
        os.makedirs(SIGNALS_DIR)
    
    # Submuestrear 10 electrodos de ECG y EGM para toda la señal
    num_electrodos_ecg = X_np.shape[1]
    num_electrodos_egm = y_np.shape[1]
    
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    if num_electrodos_ecg > 10:
        indices_ecg = np.random.choice(num_electrodos_ecg, 10, replace=False)
    else:
        indices_ecg = np.arange(num_electrodos_ecg)
    
    if num_electrodos_egm > 10:
        indices_egm = np.random.choice(num_electrodos_egm, 10, replace=False)
    else:
        indices_egm = np.arange(num_electrodos_egm)
    
    # Extraer los electrodos seleccionados para toda la serie temporal
    X_submuestreado = X_np[:, indices_ecg]
    y_submuestreado = y_np[:, indices_egm]
    
    # Combinar las señales subsampleadas de ECG y EGM
    signals_combined = np.hstack((X_submuestreado, y_submuestreado))
    
    # Crear una figura con una cuadrícula de 4 filas y 5 columnas
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    # Graficar cada señal en la cuadrícula
    for i, ax in enumerate(axes):
        if i < signals_combined.shape[1]:
            ax.plot(signals_combined[:, i])
            if i < 10:
                ax.set_title(f'ECG Electrodo {indices_ecg[i]}')
            else:
                ax.set_title(f'EGM Electrodo {indices_egm[i - 10]}')
            ax.set_xlabel('Muestras')
            ax.set_ylabel('Amplitud')
        else:
            ax.axis('off')  # Ocultar subplots adicionales si los hay
    
    # Ajustar el diseño para evitar superposición
    plt.tight_layout()
    
    # Guardar la figura
    image_filename = os.path.join(SIGNALS_DIR, 'ecg_egm_signals_grid.png')
    plt.savefig(image_filename, dpi=300)
    plt.close()
    print(f"Imagen guardada en: {image_filename}")
    
    # Mostrar las dimensiones de las señales submuestreadas
    print(f"Tamaño de X_submuestreado: {X_submuestreado.shape}")
    print(f"Tamaño de y_submuestreado: {y_submuestreado.shape}")
    
    # Crear un tensor para almacenar los escalogramas de todos los electrodos
    # Dimensiones: [num_electrodos, num_escalas, num_muestras]
    num_escalas = 128
    scales = np.arange(1, num_escalas + 1)
    num_muestras = X_np.shape[0]
    
    # Inicializar tensores para ECG y EGM
    tensor_ecg = np.zeros((num_electrodos_ecg, num_escalas, num_muestras), dtype=np.float32)
    tensor_egm = np.zeros((num_electrodos_egm, num_escalas, num_muestras), dtype=np.float32)
    
    # Mostrar las dimensiones iniciales de los tensores
    print(f"Tamaño inicial del tensor ECG: {tensor_ecg.shape}")
    print(f"Tamaño inicial del tensor EGM: {tensor_egm.shape}")
    
    # Calcular y almacenar los escalogramas para todos los electrodos de ECG
    print("Calculando escalogramas para ECG...")
    for i in range(num_electrodos_ecg):
        signal_ecg = X_np[:, i]
        coefficients, frequencies = calcular_cwt(signal_ecg, scales, wavelet_name, freq)
        tensor_ecg[i] = np.abs(coefficients)
        print(f"Electrodo ECG {i}: Dimensiones de los coeficientes CWT {coefficients.shape}")
    
    # Calcular y almacenar los escalogramas para todos los electrodos de EGM
    print("Calculando escalogramas para EGM...")
    for i in range(num_electrodos_egm):
        signal_egm = y_np[:, i]
        coefficients, frequencies = calcular_cwt(signal_egm, scales, wavelet_name, freq)
        tensor_egm[i] = np.abs(coefficients)
        print(f"Electrodo EGM {i}: Dimensiones de los coeficientes CWT {coefficients.shape}")
    
    # Mostrar el tamaño de los tensores antes de almacenarlos
    tensor_ecg_size_bytes = tensor_ecg.nbytes
    tensor_ecg_size_mb = tensor_ecg_size_bytes / (1024 * 1024)
    print(f"Tamaño del tensor ECG en memoria: {tensor_ecg_size_mb:.2f} MB")
    
    tensor_egm_size_bytes = tensor_egm.nbytes
    tensor_egm_size_mb = tensor_egm_size_bytes / (1024 * 1024)
    print(f"Tamaño del tensor EGM en memoria: {tensor_egm_size_mb:.2f} MB")
    
    # Guardar los tensores en un archivo NPZ comprimido
    tensor_filename = os.path.join(TENSOR_DIR, f'{DATASET}_tensor.npz')
    np.savez_compressed(tensor_filename, ecg=tensor_ecg, egm=tensor_egm, frequencies=frequencies, time=np.arange(num_muestras)/freq)
    print(f"Tensores guardados en: {tensor_filename}")
    
    # Información final
    print(f"Dimensiones finales del tensor ECG: {tensor_ecg.shape}")
    print(f"Dimensiones finales del tensor EGM: {tensor_egm.shape}")
    
    # Calcular una señal ECG representativa como la media de todos los electrodos
    ecg_signal = np.mean(X_np, axis=1)

    
    # Aplicar el algoritmo de Pan-Tompkins
    r_peaks = pan_tompkins_detector(ecg_signal, freq)
    print(f"Número de picos R detectados: {len(r_peaks)}")
    
    peaks_filename = os.path.join(RESULTS_DIR, 'ecg_r_peaks_mean.npy')
    np.save(peaks_filename, r_peaks)
    print(f"Picos R guardados en: {peaks_filename}")


    
    # Calcular los intervalos RR (diferencias entre picos R consecutivos)
    rr_intervals = np.diff(r_peaks) / freq  # Convertir a segundos
    print(f"Intervalos RR (primeros 10): {rr_intervals[:10]}")
    
    rr_filename = os.path.join(RESULTS_DIR, 'ecg_rr_intervals_mean.npy')
    np.save(rr_filename, rr_intervals)
    print(f"Intervalos RR guardados en: {rr_filename}")

    
    # Graficar la señal y los picos detectados
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_signal, label='ECG Signal')
    plt.plot(r_peaks, ecg_signal[r_peaks], 'ro', label='R Peaks')
    plt.title('ECG Signal with Detected R Peaks - Señal Promedio')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.tight_layout()
    # Guardar la figura
    fig_filename = os.path.join(RESULTS_DIR, 'ecg_r_peaks_mean.png')
    plt.savefig(fig_filename, dpi=300)
    plt.close()
    print(f"Figura guardada en: {fig_filename}")
    
    # Imprimir estadísticas de los intervalos RR
    print(f"Media de intervalos RR: {np.mean(rr_intervals):.4f} s")
    print(f"Desviación estándar de intervalos RR: {np.std(rr_intervals):.4f} s")
    
    # Guardar un segmento de señales de ECG y EGM para un tiempo fijo, subsampleando 10 electrodos de cada uno
    # Parámetros para seleccionar un segmento de tiempo fijo
    tiempo_inicial = 0  # En segundos
    duracion = 10  # Duración del segmento en segundos

    # Calcular los índices de inicio y fin
    muestra_inicio = int(tiempo_inicial * freq)
    muestra_fin = int((tiempo_inicial + duracion) * freq)

    # Asegurarse de que los índices están dentro del rango de las señales
    muestra_fin = min(muestra_fin, X_np.shape[0], y_np.shape[0])

    # Seleccionar el segmento de tiempo fijo
    X_segmento = X_np[muestra_inicio:muestra_fin, :]
    y_segmento = y_np[muestra_inicio:muestra_fin, :]

    # Submuestrear (seleccionar) 10 electrodos de cada uno
    if num_electrodos_ecg > 10:
        indices_ecg_seg = np.random.choice(num_electrodos_ecg, 10, replace=False)
    else:
        indices_ecg_seg = np.arange(num_electrodos_ecg)

    if num_electrodos_egm > 10:
        indices_egm_seg = np.random.choice(num_electrodos_egm, 10, replace=False)
    else:
        indices_egm_seg = np.arange(num_electrodos_egm)

    # Seleccionar los electrodos
    X_submuestreado_seg = X_segmento[:, indices_ecg_seg]
    y_submuestreado_seg = y_segmento[:, indices_egm_seg]

    # Guardar las señales seleccionadas en un archivo .mat
    output_filename = os.path.join(SIGNALS_DIR, f'{DATASET}_segmento_{tiempo_inicial}s_{duracion}s.mat')
    sio.savemat(output_filename, {
        'ecg': X_submuestreado_seg,
        'egm': y_submuestreado_seg,
        'freq': freq,
        'indices_ecg': indices_ecg_seg,
        'indices_egm': indices_egm_seg,
        'tiempo_inicial': tiempo_inicial,
        'duracion': duracion
    })
    print(f"Señales guardadas en: {output_filename}")

    # Información sobre las señales guardadas
    print(f"Tamaño de X_submuestreado_seg: {X_submuestreado_seg.shape}")
    print(f"Tamaño de y_submuestreado_seg: {y_submuestreado_seg.shape}")
    print(f"Electrodos ECG seleccionados para el segmento: {indices_ecg_seg}")
    print(f"Electrodos EGM seleccionados para el segmento: {indices_egm_seg}")
