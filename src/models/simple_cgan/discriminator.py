import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, label_emb_dim, dropout_rate=0.0):
        super(ConditionalDiscriminator, self).__init__()
        # Embedding para la etiqueta
        self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim =label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(input_dim + label_emb_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (batch_size, input_dim) => señal real o generada
        label_vec = self.label_emb(labels)  # (batch_size, label_emb_dim)
        concat = torch.cat([x, label_vec], dim=1)  # (batch_size, input_dim + label_emb_dim)
        out = self.model(concat)  # (batch_size, 1)
        return out