from torch import nn
import numpy as np


class CacheReplacementNN(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(CacheReplacementNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
