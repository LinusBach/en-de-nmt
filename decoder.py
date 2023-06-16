import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p, max_length, batch_size=1, batch_first=False):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, dropout=dropout_p,
                          batch_first=batch_first)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, features, hidden, encoder_outputs):
        embedded = self.embedding(features).view(self.batch_size, 1, -1)
        embedded = self.dropout(embedded)

        # print(embedded[:, 0].shape, hidden[0].shape, encoder_outputs.shape)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0], hidden[0]), 1)), dim=1)
        # print(attn_weights.shape, encoder_outputs.shape)
        # torch.Size([1, 30]) torch.Size([1, 1, 30]) torch.Size([30, 100]) torch.Size([1, 30, 100])
        # print(attn_weights.unsqueeze(1).shape, encoder_outputs[0].unsqueeze(0).shape)
        # print(attn_weights.unsqueeze(1).shape, encoder_outputs.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        print("39", embedded.shape, attn_applied.shape)
        output = torch.cat((embedded, attn_applied), 1)
        print("41", output.shape)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        print(output.shape, hidden.shape)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device="cpu"):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)
