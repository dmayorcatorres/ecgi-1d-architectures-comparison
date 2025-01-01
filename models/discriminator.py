import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, debug=False):
        """
        Discriminador de datos basado en redes neuronales.

        Args:
            input_dim (int): Dimensión de entrada.
            latent_dim (int): Dimensión de salida de LSTM.
            debug (bool): Modo debug para imprimir detalles.
        """
        super(Discriminator, self).__init__()
        
        self.debug = debug  # Modo Debug
        
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

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x):
        """
        Paso hacia adelante en el discriminador.

        Args:
            x (torch.Tensor): Tensor de entrada con tamaño [batch_size, input_dim].
        """
        if self.debug:
            print("🔄 **Entrada al Discriminador (antes de Unflatten):**", x.shape)
        
        x = self.model(x)
        
        if self.debug:
            print("✅ **Después de model:**", x.shape)
        
        x = x.unsqueeze(1)
        
        if self.debug:
            print("✅ **Después de unsqueeze:**", x.shape)
        
        x, _ = self.lstm(x)
        
        if self.debug:
            print("✅ **Después de LSTM:**", x.shape)
        
        output = self.fc(x[:, -1, :])
        
        if self.debug:
            print("✅ **Después de FC:**", output.shape)
        
        return output
