import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.gru3 = nn.GRU(hidden_size, hidden_size)

    def forward(self, features, hidden):
        embedded = self.embedding(features).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru1(output, hidden)
        output, hidden = self.gru2(output, hidden)
        output, hidden = self.gru3(output, hidden)
        return output, hidden

    def init_hidden(self, device="cpu"):
        return torch.zeros(1, 1, self.hidden_size, device=device)
