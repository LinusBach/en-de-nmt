import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, features, hidden):
        embedded = self.embedding(features).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, device="cpu"):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
