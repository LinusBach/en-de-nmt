"""
This script defines the Encoder class for the Seq2Seq model. The Encoder class uses a recurrent neural network (RNN)
to process the input sentences. It converts the input sentences into a context vector that captures the semantic
information of the sentences. This context vector is then used by the decoder to generate the output sentences.
"""


import torch
import torch.nn as nn
from typing import Tuple


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_p: float):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_p)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(features).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, device: torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
