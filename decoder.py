import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout_p=0.1, max_length=None):
        super(AttnDecoderRNN, self).__init__()
        assert max_length is not None
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, features, hidden, cell, encoder_outputs):
        embedded = self.embedding(features).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        print(output.shape, hidden.shape)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, (hidden, cell), attn_weights

    def init_hidden(self, device="cpu"):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device),\
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
