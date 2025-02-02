import sys
import os
import datetime
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from scipy.stats import entropy, wasserstein_distance
from scipy.signal import welch
import pywt
from fastdtw import fastdtw

# Añadir la raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs import load_config
from src.utils.logger import setup_logging
from src.data.data_loader import load_data
from torch.utils.data import TensorDataset
from src.models.simple_cgan import ConditionalGenerator

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


class ECGMetricsComparator:
    def __init__(self, real_matrix: torch.Tensor, generated_matrix: torch.Tensor):
        assert real_matrix.shape == generated_matrix.shape, "Las matrices deben tener la misma forma"
        self.real_matrix = real_matrix.detach().cpu().numpy()
        self.generated_matrix = generated_matrix.detach().cpu().numpy()

    def compute_mse(self):
        return np.mean((self.real_matrix - self.generated_matrix) ** 2)

    def compute_ssim(self):
        data_range = max(self.real_matrix.max(), self.generated_matrix.max()) - min(self.real_matrix.min(), self.generated_matrix.min())
        ssim_values = [ssim(self.real_matrix[:, i], self.generated_matrix[:, i], data_range=data_range) for i in range(self.real_matrix.shape[1])]
        return np.mean(ssim_values)

    def compute_cosine_similarity(self):
        cos_sim = [1 - cosine(self.real_matrix[:, i], self.generated_matrix[:, i]) for i in range(self.real_matrix.shape[1])]
        return np.mean(cos_sim)

    def compute_dtw_distance(self):
        dtw_distances = [fastdtw(self.real_matrix[:, i], self.generated_matrix[:, i])[0] for i in range(self.real_matrix.shape[1])]
        return np.mean(dtw_distances)

    def compute_total_variation_distance(self):
        return np.mean(np.abs(self.real_matrix - self.generated_matrix))

    def compute_wasserstein_distance(self):
        wasserstein_distances = [wasserstein_distance(self.real_matrix[:, i], self.generated_matrix[:, i]) for i in range(self.real_matrix.shape[1])]
        return np.mean(wasserstein_distances)

    def compute_spectral_entropy(self):
        def entropy_from_signal(signal):
            f, Pxx = welch(signal, fs=1000, nperseg=256)
            Pxx = Pxx / np.sum(Pxx)
            return entropy(Pxx)

        real_entropies = [entropy_from_signal(self.real_matrix[:, i]) for i in range(self.real_matrix.shape[1])]
        gen_entropies = [entropy_from_signal(self.generated_matrix[:, i]) for i in range(self.generated_matrix.shape[1])]
        return np.mean(np.abs(np.array(real_entropies) - np.array(gen_entropies)))

    def compute_wavelet_energy_distance(self):
        def wavelet_energy(signal):
            coeffs, _ = pywt.cwt(signal, scales=np.arange(1, 65), wavelet='morl')
            return np.sum(np.abs(coeffs))

        real_energies = [wavelet_energy(self.real_matrix[:, i]) for i in range(self.real_matrix.shape[1])]
        gen_energies = [wavelet_energy(self.generated_matrix[:, i]) for i in range(self.generated_matrix.shape[1])]
        return np.mean(np.abs(np.array(real_energies) - np.array(gen_energies)))

    def compute_all_metrics(self):
        return {
            "MSE": self.compute_mse(),
            "SSIM": self.compute_ssim(),
            "Cosine Similarity": self.compute_cosine_similarity(),
            "DTW Distance": self.compute_dtw_distance(),
            "Total Variation Distance": self.compute_total_variation_distance(),
            "Wasserstein Distance": self.compute_wasserstein_distance(),
            "Spectral Entropy Difference": self.compute_spectral_entropy(),
            "Wavelet Energy Distance": self.compute_wavelet_energy_distance()
        }

    def plot_sampled_signals(self, exp_dir, num_samples=12):
        num_electrodes = self.real_matrix.shape[1]
        sampled_indices = np.random.choice(num_electrodes, num_samples, replace=False)

        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        fig.suptitle("Comparación de señales originales y generadas", fontsize=16)

        for i, ax in enumerate(axes.flatten()):
            if i >= len(sampled_indices):
                ax.axis("off")
                continue
            
            idx = sampled_indices[i]
            ax.plot(self.real_matrix[:, idx], label=f'Electrodo {idx+1} (Real)', color='blue', alpha=0.7)
            ax.plot(self.generated_matrix[:, idx], label=f'Electrodo {idx+1} (Generado)', color='red', linestyle='dashed', alpha=0.7)
            ax.set_title(f"Electrodo {idx+1}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Guardar la gráfica en PNG
        plot_path = os.path.join(exp_dir, "electrode_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfica guardada en: {plot_path}")


if __name__ == "__main__":
    # Directorio del experimento
    base_dir = "/workspace/pytorch_project/syntetic_data"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{now_str}")
    os.makedirs(exp_dir, exist_ok=True)

    # Configuración del logger
    logger = setup_logging(exp_dir)
    logger.info(f"Creando carpeta de experimento: {exp_dir}")

    # Configurar el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")

    # Cargar los datos
    mat_file = os.path.abspath("/workspace/pytorch_project/data/raw/kitpvc/kitpvc_data.mat")
    data, tags_np = load_data(mat_file, logger)

    batch_size = 16
    latent_dim = 64
    dropout = 0.1

    # Convertir etiquetas a LongTensor
    tags = torch.from_numpy(tags_np).long().to(device)
    egm = torch.tensor(data['egm'], dtype=torch.float32).to(device)
    logger.info(f"Dimensiones de EGM: {egm.shape}")

    # Generar muestras falsas
    egm_gen = torch.randn(egm.shape[0], latent_dim, device=device)
    input_dim = egm.shape[1]
    num_classes = int(tags.max().item() + 1)
    label_emb_dim = 1

    # Cargar modelo generador
    generator = ConditionalGenerator(latent_dim, num_classes, label_emb_dim, input_dim, dropout)
    model_path = "/workspace/pytorch_project/experiments/experiment_20250126_024407/generator_best.pt"
    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    logger.info("Modelo generador cargado y puesto en modo evaluación.")

    # Generar señales sintéticas
    fake_samples = generator(egm_gen, tags)
    logger.info(f"Dimensiones de las muestras generadas: {fake_samples.shape}")

    # Comparar métricas
    comparator = ECGMetricsComparator(egm, fake_samples)
    metrics = comparator.compute_all_metrics()

    # Guardar métricas en JSON
    metrics_path = os.path.join(exp_dir, "metrics.json")
    metrics_float = {key: float(value) for key, value in metrics.items()}

    with open(metrics_path, "w") as f:
        json.dump(metrics_float, f, indent=4)
    logger.info(f"Métricas guardadas en: {metrics_path}")

    # Generar y guardar gráfico
    comparator.plot_sampled_signals(exp_dir)
