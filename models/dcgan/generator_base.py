import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, debug=False):
        """
        Generador de datos basado en redes neuronales.

        Args:
            latent_dim (int): Dimensión del espacio latente.
            output_dim (int): Dimensión de la salida esperada.
            debug (bool): Modo debug para imprimir detalles.
        """
        super(Generator, self).__init__()
        
        self.debug = debug  # Modo Debug
        
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
        
        # Calcular la dimensión para LSTM
        with torch.no_grad():
            dummy_input = torch.zeros(1, latent_dim)
            self.model.eval()
            lstm_input_size = self.model(dummy_input).shape[-1]
            self.model.train()

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=output_dim,
            num_layers=2,
            batch_first=True
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Paso hacia adelante en el generador.

        Args:
            z (torch.Tensor): Tensor de entrada con tamaño [batch_size, latent_dim].
        """
        if self.debug:
            print("🔄 **Entrada al Generador (antes de Unflatten):**", z.shape)
        
        x = self.model(z)
        
        if self.debug:
            print("✅ **Después de model:**", x.shape)
        
        x, _ = self.lstm(x)
        
        if self.debug:
            print("✅ **Después de LSTM:**", x.shape)

        return self.tanh(x).squeeze(1)
