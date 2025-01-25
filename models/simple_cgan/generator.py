# -----------------------------------------------------------------------------
# 2) MODELOS: simple_cgan.py (ConditionalGenerator, ConditionalDiscriminator)
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, label_emb_dim, output_dim, dropout_rate=0.0):
        super(ConditionalGenerator, self).__init__()
        # Embedding para la etiqueta, de dimension label_emb_dim
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (batch_size, latent_dim)
        # labels: (batch_size,) con valores int [0..num_classes-1]
        label_vec = self.label_emb(labels)  # (batch_size, label_emb_dim)
        concat = torch.cat([z, label_vec], dim=1)  # (batch_size, latent_dim + label_emb_dim)
        out = self.model(concat)
        return out