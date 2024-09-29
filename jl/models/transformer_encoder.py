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
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = x.transpose(
            0, 1
        )  # Transformer expects input of shape (seq_len, batch_size)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate over the sequence length
        x = self.fc_out(x)
        return x


class JointTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dim,
        out_dim,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        max_len=100,
    ):
        super(JointTransformerEncoder, self).__init__()

        self.embed_dim = embed_dim

        # Embedding layers for each feature
        self.cache_pc_embedding = nn.Embedding(input_dims[0], embed_dim)
        self.prefetch_pc_embedding = nn.Embedding(input_dims[1], embed_dim)
        self.prefetch_page_embedding = nn.Embedding(input_dims[2], embed_dim)
        self.prefetch_offset_embedding = nn.Embedding(input_dims[3], embed_dim)

        # Positional encodings
        self.cache_pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        self.prefetch_pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

        # Separate transformer encoders for cache and prefetcher
        cache_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.cache_transformer_encoder = nn.TransformerEncoder(
            cache_encoder_layer, num_layers
        )

        prefetch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.prefetch_transformer_encoder = nn.TransformerEncoder(
            prefetch_encoder_layer, num_layers
        )

        # Fully connected output layer
        self.fc_out = nn.Linear(embed_dim, out_dim)

    def forward(self, cache_pc, prefetch_pc, prefetch_page, prefetch_offset):
        """
        Inputs:
        - cache_pc: Tensor of shape (cache_batch_size, cache_seq_len)
        - prefetch_pc, prefetch_page, prefetch_offset: Tensors of shape (prefetch_batch_size, prefetch_seq_len)
        """
        ## Cache sequence processing
        cache_pc_emb = self.cache_pc_embedding(cache_pc) * math.sqrt(self.embed_dim)
        cache_emb = self.cache_pos_encoder(
            cache_pc_emb.transpose(0, 1)
        )  # (cache_seq_len, batch_size, embed_dim)
        cache_output = self.cache_transformer_encoder(cache_emb)
        cache_output = cache_output.mean(dim=0)  # Aggregating over the cache sequence

        ## Prefetcher sequence processing
        prefetch_pc_emb = self.prefetch_pc_embedding(prefetch_pc) * math.sqrt(
            self.embed_dim
        )
        prefetch_page_emb = self.prefetch_page_embedding(prefetch_page) * math.sqrt(
            self.embed_dim
        )
        prefetch_offset_emb = self.prefetch_offset_embedding(
            prefetch_offset
        ) * math.sqrt(self.embed_dim)

        # Sum or concatenate prefetch features (here we use summation)
        prefetch_combined_emb = (
            prefetch_pc_emb + prefetch_page_emb + prefetch_offset_emb
        )
        prefetch_emb = self.prefetch_pos_encoder(
            prefetch_combined_emb.transpose(0, 1)
        )  # (prefetch_seq_len, batch_size, embed_dim)

        prefetch_output = self.prefetch_transformer_encoder(prefetch_emb)

        # Aggregating over the prefetch sequence
        prefetch_output = prefetch_output.mean(dim=0)

        # Combine cache and prefetch outputs
        # Can also concatenate or use other methods
        combined_output = cache_output + prefetch_output

        # Pass through final output layer
        output = self.fc_out(combined_output)

        return output


class PrefetchTransformerEncoder(nn.Module):
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
        super(PrefetchTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim

        self.prefetch_pc_embedding = nn.Embedding(input_dim[0], embed_dim)
        self.prefetch_page_embedding = nn.Embedding(input_dim[1], embed_dim)
        self.prefetch_offset_embedding = nn.Embedding(input_dim[2], embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, out_dim)

    def forward(self, prefetch_pc, prefetch_page, prefetch_offset):
        """
        Inputs:
        - prefetch_pc, prefetch_page, prefetch_offset: Tensors of shape (prefetch_batch_size, prefetch_seq_len)
        """
        prefetch_pc_emb = self.prefetch_pc_embedding(prefetch_pc) * math.sqrt(
            self.embed_dim
        )
        prefetch_page_emb = self.prefetch_page_embedding(prefetch_page) * math.sqrt(
            self.embed_dim
        )
        prefetch_offset_emb = self.prefetch_offset_embedding(
            prefetch_offset
        ) * math.sqrt(self.embed_dim)

        # Sum or concatenate prefetch features (here we use summation)
        prefetch_combined_emb = (
            prefetch_pc_emb + prefetch_page_emb + prefetch_offset_emb
        )
        prefetch_emb = self.pos_encoder(
            prefetch_combined_emb.transpose(0, 1)
        )  # (prefetch_seq_len, batch_size, embed_dim)

        prefetch_output = self.transformer_encoder(prefetch_emb)

        # Aggregating over the prefetch sequence
        prefetch_output = prefetch_output.mean(dim=0)

        # Pass through final output layer
        output = self.fc_out(prefetch_output)

        return output
