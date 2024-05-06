import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, drop_out=0.1):
        super(ContrastiveEncoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.drop_out = drop_out

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=out_dim, num_heads=1, dropout=drop_out)
        # nn.init.xavier_uniform_(self.attention_weight_vector.data)

    def apply_attention(self, x):
        # Compute attention scores using a simple dot product, and softmax to form a probability distribution
        attention_scores = torch.matmul(
            x, self.attention_weight_vector
        )  # [batch size, seq_length]
        attention_scores = F.softmax(attention_scores, dim=1).unsqueeze(
            2
        )  # [batch size, seq_length, 1]

        # Weighted sum of sequence elements
        attended = torch.sum(x * attention_scores, dim=1)  # [batch size, out_dim]
        return attended

    def forward(self, x):
        original_dim = x.dim()

        if original_dim == 3:
            # Input shape [batch size, seq_length, input_size]
            batch_size, seq_length, embed_size = x.shape
            x = x.view(-1, embed_size)

        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.fc2(x)

        if original_dim == 3:
            x = x.view(seq_length, batch_size, -1)
            # Apply attention
            x, _ = self.attention(x, x, x)  # Self-attention
            x = x.transpose(0, 1).contiguous()  # Back to [batch size, seq_length, out_dim]
            x = x.mean(dim=1)  # Mean pooling over sequence
            # x = x.view(batch_size, seq_length, -1)
            # x = self.apply_attention(x)

        return x
