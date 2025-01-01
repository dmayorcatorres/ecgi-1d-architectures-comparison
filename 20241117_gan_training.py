import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.amp import autocast, GradScaler

# Variable para controlar el uso de Google Drive o rutas locales
USE_DRIVE = False

# Configurar rutas según el valor de USE_DRIVE
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    DATASET = "kitpvc"
    DATA_PATH = f"/content/drive/MyDrive/UGR/deeplearning_ecgi/data/{DATASET}/"
    MODEL_PATH = f"/content/drive/MyDrive/UGR/deeplearning_ecgi/models/pytorch/{DATASET}/"
    PLOT_PATH = f"/content/drive/MyDrive/UGR/deeplearning_ecgi/plots/{DATASET}/"
else:
    # Establecer BASE_PATH como la ruta absoluta de la carpeta que contiene el script
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # Esto obtiene la ruta de la carpeta del script
    DATASET = "kitpvc"
    DATA_PATH = os.path.join(BASE_PATH, f"data/raw/{DATASET}/")
    MODEL_PATH = os.path.join(BASE_PATH, f"models/pytorch/{DATASET}/")
    PLOT_PATH = os.path.join(BASE_PATH, f"plots/{DATASET}/")

# Crear directorios si no existen
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Optimización para convoluciones

# Funciones auxiliares
def tanh_normalization(y):
    return 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

def load_data():
    data = sio.loadmat(os.path.join(DATA_PATH, f"{DATASET}_data.mat"))
    return data['ecg'], data['egm'], np.array(data['tags']), data['A']

# Cargar datos
X, y, label, A = load_data()

# Normalizar y preparar datos
y = tanh_normalization(y)
y = torch.tensor(y, dtype=torch.float32)

num_rows, num_cols = y.shape
batch_size = 8
epochs = 1000
latent_dim = 32

dataset = TensorDataset(y)

# Definición del Generador
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=False),
            nn.Unflatten(1, (latent_dim, 1)),
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(8, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Unflatten(1, (1, -1)),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, latent_dim)
            self.model.eval()
            lstm_input_size = self.model(dummy_input).shape[-1]
            self.model.train()

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=output_dim, num_layers=2, batch_first=True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.model(z)
        x, _ = self.lstm(x)
        return self.tanh(x).squeeze(1)

# Definición del Discriminador
class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.3),
            nn.MaxPool1d(4),
            nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim)
            self.model.eval()
            lstm_input_size = self.model(dummy_input).shape[-1]
            self.model.train()

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=latent_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# Inicialización de modelos y optimizadores
generator = Generator(latent_dim, num_cols).to(device)
discriminator = Discriminator(num_cols, latent_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.00005)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

real_label = 0.9
fake_label = 0.0

G_losses = []
D_losses = []
save_epochs = [250, 500, 750, 1000]

scaler_G = GradScaler()
scaler_D = GradScaler()

def train():
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        for data in dataloader:
            real_samples = data[0].to(device)
            batch_size_curr = real_samples.size(0)

            # Imprimir tamaños de las variables
            print(f"🔍 Tamaño de real_samples: {real_samples.shape}")
            print(f"🔍 Tamaño de batch_size_curr: {batch_size_curr}")

            ############################
            # Entrenamiento del Discriminador
            ############################
            optimizer_D.zero_grad()
            with autocast(device_type='cuda'):  # Especificar el dispositivo
                labels_real = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)
                output_real = discriminator(real_samples).view(-1)
                print(f"🔍 Tamaño salida real: {output_real.shape}, Tamaño etiquetas reales: {labels_real.shape}")
                loss_D_real = criterion(output_real, labels_real)

                noise = torch.randn(batch_size_curr, latent_dim, device=device)
                fake_samples = generator(noise)
                labels_fake = torch.full((batch_size_curr,), fake_label, dtype=torch.float, device=device)
                output_fake = discriminator(fake_samples.detach()).view(-1)
                print(f"🔍 Tamaño salida falsa: {output_fake.shape}, Tamaño etiquetas falsas: {labels_fake.shape}")
                loss_D_fake = criterion(output_fake, labels_fake)

                loss_D = loss_D_real + loss_D_fake

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            ############################
            # Entrenamiento del Generador
            ############################
            optimizer_G.zero_grad()
            with autocast(device_type='cuda'):  # Especificar el dispositivo
                labels_g = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)
                output_g = discriminator(fake_samples).view(-1)
                loss_G = criterion(output_g, labels_g)

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        if epoch in save_epochs:
            torch.save(generator.state_dict(), os.path.join(MODEL_PATH, f"generator_epoch_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_PATH, f"discriminator_epoch_{epoch}.pt"))

        # Mostrar resultados
        if epoch % 50 == 0 or epoch == epochs:
            with torch.no_grad():
                samples = generator(torch.randn(16, latent_dim, device=device)).cpu().numpy()
                plt.figure(figsize=(10, 5))
                for i in range(16):
                    plt.subplot(4, 4, i + 1)
                    plt.plot(samples[i])
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_PATH, f"epoch_{epoch}.png"))
                plt.close()

train()