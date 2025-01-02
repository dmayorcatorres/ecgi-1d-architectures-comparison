import torch
import torch.nn as nn

class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
