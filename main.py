import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import *
from evaluate import evaluate
from dataloader import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5

epochs = 10

input_lang, output_lang, pairs = prepare_data('data/train.en', 'data/train.de', 100)
print(f'number of pairs: {len(pairs)}')

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)

# TODO add pairs
train_iters(encoder1, attn_decoder1, pairs, input_lang, output_lang, epochs, print_every=50, max_length=MAX_LENGTH,
            device=device)
evaluate(encoder1, attn_decoder1, "I am cold.", input_lang, output_lang, max_length=MAX_LENGTH, device=device)
