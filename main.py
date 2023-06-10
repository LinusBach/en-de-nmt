import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import *
from evaluate import evaluate
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

teacher_forcing_ratio = 1
n_iters = 20
n_samples = 100
lr = 2e-4
hidden_size = 1024
dropout = 0.1

print_every = 10

input_lang, output_lang, pairs = prepare_data('data/train.en', 'data/train.de', n_samples)
print(f'number of pairs: {len(pairs)}')

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout, max_length=MAX_LENGTH).to(device)

train_iters(encoder1, attn_decoder1, pairs, input_lang, output_lang, n_iters, print_every=print_every, learning_rate=lr,
            max_length=MAX_LENGTH, device=device)
# evaluate(encoder1, attn_decoder1, "announcement", input_lang, output_lang, max_length=MAX_LENGTH, device=device)
