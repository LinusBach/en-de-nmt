import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p, batch_size=1, batch_first=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=batch_first)

    def forward(self, features, hidden, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # print(features.shape, hidden.shape)
        embedded = self.embedding(features).view(1, batch_size, -1)
        embedded = self.dropout(embedded)
        print("encoder - 25", embedded.shape, torch.sum(embedded, dim=2))
        # print(embedded.shape, hidden.shape)
        output, hidden = self.gru(embedded, hidden)
        print("encoder", hidden.shape, hidden)
        # output = F.normalize(output, dim=2)
        # hidden = F.normalize(hidden, dim=2)
        return output, hidden

    def init_hidden(self, batch_size=None, device="cpu"):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
