from torch import nn
import numpy as np

from models.contrastive_encoder import ContrastiveEncoder


class CacheReplacementNN(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNN, self).__init__()
        if contrastive_encoder is not None:
            for param in contrastive_encoder.parameters():
                param.requires_grad = False

            self.network = nn.Sequential(
                contrastive_encoder,
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.network = nn.Sequential(
                ContrastiveEncoder(num_features, hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, 1),
            )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x)
