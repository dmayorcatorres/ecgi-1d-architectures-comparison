import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout_rate):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
