"""
This script defines the Decoder class for the Seq2Seq model. The Decoder uses a recurrent neural network (RNN)
and an attention mechanism to generate the output sentences from the context vector provided by the encoder.
The attention mechanism allows the decoder to focus on different parts of the input sentence at each step of the
output sentence generation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class AttnDecoderRNN(nn.Module):
    def __init__(self, model_variant: str, hidden_size: int, output_size: int, num_layers: int, dropout_p: float,
                 max_length: int):
        super(AttnDecoderRNN, self).__init__()

        self.model_variant = model_variant
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if model_variant == "gru":
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, dropout=dropout_p)
        elif model_variant == "lstm":
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, features: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                encoder_outputs: torch.Tensor) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        if self.model_variant == "lstm":
            cell = hidden[1]
            hidden = hidden[0]

        embedded = self.embedding(features).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        if self.model_variant == "gru":
            output, hidden = self.rnn(output, hidden)
        elif self.model_variant == "lstm":
            output, hidden = self.rnn(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device: torch.device = "cpu") -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.model_variant == "gru":
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        else:
            return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_layers, 1, self.hidden_size, device=device))
