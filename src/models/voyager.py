import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming 'voyager.losses' and 'voyager.metrics' are custom modules and need to be rewritten for PyTorch.
from voyager.losses_pytorch import HierarchicalSequenceLoss, HierarchicalCrossEntropyWithLogitsLoss
from voyager.metrics_pytorch import *

def get_model(model_name):
    if model_name == 'voyager':
        return Voyager
    elif model_name == 'offset_voyager':
        return OffsetVoyager
    elif model_name == 'offset_lstm':
        return OffsetLSTM

class Stateless:
    '''
    Stateless dropout implementation for reproducible dropout
    '''
    @staticmethod
    def dropout_input(x, rate, seed, training):
        if not training:
            return x
        torch.manual_seed(seed)
        dropout_mask = x.new_empty(x.shape).bernoulli_(1 - rate)
        return x * dropout_mask

class Voyager(nn.Module):

    def __init__(self, config, pc_vocab_size, page_vocab_size):
        super(Voyager, self).__init__()
        self.config = config
        self.steps_per_epoch = config.steps_per_epoch
        self.step = 0
        self.epoch = 1

        self.offset_size = (1 << config.offset_bits)
        self.pc_embed_size = config.pc_embed_size
        self.page_embed_size = config.page_embed_size
        self.offset_embed_size = config.page_embed_size * config.num_experts
        self.lstm_size = config.lstm_size
        self.num_layers = config.lstm_layers
        self.sequence_length = config.sequence_length
        self.pc_vocab_size = pc_vocab_size
        self.page_vocab_size = page_vocab_size
        self.batch_size = config.batch_size
        self.sequence_loss = config.sequence_loss
        self.dropout = config.lstm_dropout

        self.input_size = config.input_size

        self.init()

    def init(self):
        self.init_embed()
        self.init_mha()
        self.init_lstm()
        self.init_linear()

    def init_embed(self):
        # Embedding Layers
        self.pc_embedding = nn.Embedding(self.pc_vocab_size, self.pc_embed_size)
        self.page_embedding = nn.Embedding(self.page_vocab_size, self.page_embed_size)
        self.offset_embedding = nn.Embedding(self.offset_size, self.offset_embed_size)

    def init_mha(self):
        # Page-Aware Offset Embedding
        self.mha = nn.MultiheadAttention(
            embed_dim=self.page_embed_size,
            num_heads=1,
        )

    def init_linear(self):
        # Linear layers
        self.page_linear = nn.Linear(self.lstm_size, self.page_vocab_size)
        self.offset_linear = nn.Linear(self.lstm_size, self.page_vocab_size)

    def init_lstm(self):
        if self.sequence_loss:
            # If sequence loss is true, all layers return sequences
            self.coarse_layers = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.config.lstm_dropout,
                return_sequences=True
            )
            self.fine_layers = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.config.lstm_dropout,
                return_sequences=True
            )
        else:
            # When not returning sequences, only the last layer returns the output
            self.coarse_layers = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.config.lstm_dropout
            )
            self.fine_layers = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.config.lstm_dropout
            )

    def address_embed(self, pages, offsets):
        page_embed = self.page_embedding(pages)
        offset_embed = self.offset_embedding(offsets)

        # Compute page-aware offset embedding
        tmp_page_embed = page_embed.view(-1, self.sequence_length, 1, self.page_embed_size)
        offset_embed = offset_embed.view(-1, self.sequence_length, self.offset_embed_size // self.page_embed_size, self.page_embed_size)

        tmp_page_embed = tmp_page_embed.transpose(0, 1)  # Shape: (sequence_length, batch, 1, page_embed_size)
        offset_embed = offset_embed.transpose(0, 1)      # Shape: (sequence_length, batch, offset_embed_size // page_embed_size, page_embed_size)

        offset_embed, _ = self.mha(tmp_page_embed, offset_embed, offset_embed, need_weights=False)

        # Transpose back to original shape
        offset_embed = offset_embed.transpose(0, 1).contiguous().view(-1, self.sequence_length, self.page_embed_size)

        return page_embed, offset_embed
    
    def lstm_output(self, lstm_inputs):
        # lstm_inputs = F.dropout(lstm_inputs, p=self.config.dropout, training=training)
        coarse_out, _ = self.coarse_layers(lstm_inputs)
        fine_out, _ = self.fine_layers(lstm_inputs)

        return coarse_out, fine_out
    
    def linear(self, lstm_output):
        coarse_out, fine_out = lstm_output
        # Pass through linear layers
        coarse_logits = self.page_linear(coarse_out)
        fine_logits = self.offset_linear(fine_out)

        # Full sequence has a time dimension
        if self.sequence_loss:
            return torch.cat([coarse_logits, fine_logits], dim=2)
        else:
            return torch.cat([coarse_logits, fine_logits], dim=1)


    def forward(self, x):
        pcs, pages, offsets = x[:, :self.sequence_length], x[:, self.sequence_length:2*self.sequence_length], x[:, 2*self.sequence_length:3*self.sequence_length]

        pc_embed = self.pc_embedding(pcs)

        page_embed, offset_embed = self.address_embed(pages, offsets)

        if self.config.pc_localized and self.config.global_stream:
            pc_localized_pcs = x[:, 3 * self.sequence_length:4 * self.sequence_length]
            pc_localized_pages = x[:, 4 * self.sequence_length:5 * self.sequence_length]
            pc_localized_offsets = x[:, 5 * self.sequence_length:6 * self.sequence_length]

            # Compute embeddings
            pc_localized_pc_embed = self.pc_embedding(pc_localized_pcs)
            pc_localized_page_embed, pc_localized_offset_embed = self.address_embed(pc_localized_pages, pc_localized_offsets)

            lstm_inputs = torch.cat([pc_embed, page_embed, offset_embed,
                pc_localized_pc_embed, pc_localized_page_embed, pc_localized_offset_embed], dim=2)
        else:
            lstm_inputs = torch.cat([pc_embed, page_embed, offset_embed], dim=2)

        lstm_output = self.lstm_output(lstm_inputs)

        return self.linear(lstm_output)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    @classmethod
    def compile_model(cls, config, num_unique_pcs, num_unique_pages):
        model = cls(config, num_unique_pcs, num_unique_pages)
        # Loss and optimizer
        if config.sequence_loss:
            loss = HierarchicalSequenceLoss()
        else:
            loss = HierarchicalCrossEntropyWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        return model, loss, optimizer

# For OffsetVoyager and OffsetLSTM, similar changes need to be applied.
