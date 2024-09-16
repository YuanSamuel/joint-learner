import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, embed_dim) with positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, embed_dim)

        # Register pe as a buffer so it's saved and loaded with the model state
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch_size, embed_dim)
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        out_dim,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        max_len=100,
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len)
        """
        x = x.transpose(
            0, 1
        )  # Transformer expects input of shape (seq_len, batch_size)
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over the sequence length
        x = self.fc_out(x)
        return x
