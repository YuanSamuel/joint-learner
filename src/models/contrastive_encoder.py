import torch
import torch.nn as nn


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

    def forward(self, x):
        if x.dim() == 3:
            # Input shape [batch size, seq_length, input_size]
            batch_size, seq_length, embed_size = x.shape
            x = x.view(-1, embed_size)  # Flatten seq_length into batch size for processing
        elif x.dim() == 2:
            # Input shape [batch size, input_size]
            batch_size = x.size(0)
            embed_size = x.size(1)
        
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.fc2(x)

        if x.dim() == 2 and 'seq_length' in locals():
            x = x.view(batch_size, seq_length, -1) 

        return x
