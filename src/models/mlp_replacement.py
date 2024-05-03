from torch import nn
import numpy as np


class CacheReplacementNN(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(CacheReplacementNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x)
