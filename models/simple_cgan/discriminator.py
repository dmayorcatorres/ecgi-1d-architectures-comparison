import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes, label_emb_dim, dropout_rate):
        super(ConditionalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
