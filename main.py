import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import *
from evaluate import evaluate
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

eval_doc = "data/newstest2015"

models_dir = "models"
model_name = "6_layers_512_hidden"
plots_dir = "plots"

teacher_forcing_ratio = 1
n_iters = 500
n_samples = 100000
lr = 2e-4
hidden_size = 512
n_encoder_layers = 6
n_decoder_layers = 6
dropout = 0.1

print_every = 100
plot_every = 50
save_every = 1000

input_lang, output_lang, pairs = prepare_data('data/train.en', 'data/train.de', n_samples)
print(f'number of pairs: {len(pairs)}')

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, num_layers=n_encoder_layers).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers=n_decoder_layers, dropout_p=dropout,
                               max_length=MAX_LENGTH).to(device)

train_iters(encoder1, attn_decoder1, pairs, input_lang, output_lang, n_iters,
            print_every=print_every, plot_every=plot_every, save_every=save_every,
            learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio, max_length=MAX_LENGTH,
            device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir)
# evaluate(encoder1, attn_decoder1, "announcement", input_lang, output_lang, max_length=MAX_LENGTH, device=device)
