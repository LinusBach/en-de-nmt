"""
This script defines the Encoder class for the Seq2Seq model. The Encoder class uses a recurrent neural network (RNN)
to process the input sentences. It converts the input sentences into a context vector that captures the semantic
information of the sentences. This context vector is then used by the decoder to generate the output sentences.
"""


import torch
import torch.nn as nn
from typing import Tuple, Union


class EncoderRNN(nn.Module):
    def __init__(self, model_variant: str, input_size: int, hidden_size: int, num_layers: int, dropout_p: float):
        super(EncoderRNN, self).__init__()
        self.model_variant = model_variant
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if model_variant == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p)
        elif model_variant == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p)

    def forward(self, features: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        embedded = self.embedding(features).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self, device: torch.device = "cpu") -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.model_variant == "gru":
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        else:
            return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_layers, 1, self.hidden_size, device=device))
