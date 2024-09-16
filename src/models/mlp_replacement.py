import torch

from torch import nn
import numpy as np

from models.contrastive_encoder import ContrastiveEncoder
from models.transformer_encoder import TransformerEncoder

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

# Concat embedder features
class CacheReplacementNNConcatEmbeddings(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNNConcatEmbeddings, self).__init__()
        if contrastive_encoder is not None:
            for param in contrastive_encoder.parameters():
                param.requires_grad = False

            self.contrastive_encoder = contrastive_encoder
        else:
            self.contrastive_encoder = ContrastiveEncoder(
                num_features, hidden_dim, hidden_dim
            )

        self.network = nn.Sequential(
            nn.Linear(num_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        input_embeddings = self.contrastive_encoder(x)
        x = torch.cat((x, input_embeddings), dim=1)
        return self.network(x)

class CacheReplacementNNTransformer(nn.Module):
    def __init__(self, num_features, hidden_dim, contrastive_encoder=None):
        super(CacheReplacementNNTransformer, self).__init__()
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
                TransformerEncoder(num_features, hidden_dim, hidden_dim),
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