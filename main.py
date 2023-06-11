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
model_name = "4_layers_512_hidden_1e-4_lr"
plots_dir = "plots"
resume_training = False

n_iters = 20000
start_from_sample = 0
random_sampling = False
n_samples = 110000
patience = 20  # early stopping
patience_interval = 100
batch_first = True
batch_size = 1

teacher_forcing_ratio = 1
lr = 1e-4
hidden_size = 1024
n_encoder_layers = 4
n_decoder_layers = 4
dropout = 0.1

print_every = 100
plot_every = 1000
save_every = 1000

input_lang, output_lang, pairs = prepare_data('data/train.en', 'data/train.de', n_samples, start_from_sample)
print(f'number of pairs: {len(pairs)}')

encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers=n_encoder_layers, dropout_p=dropout,
                     batch_first=batch_first).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers=n_decoder_layers,
                              dropout_p=dropout, max_length=MAX_LENGTH, batch_first=batch_first).to(device)

if resume_training:
    encoder.load_state_dict(torch.load(os.path.join(models_dir, model_name, "encoder.pt"), map_location=device))
    attn_decoder.load_state_dict(torch.load(os.path.join(models_dir, model_name, "decoder.pt"), map_location=device))
    prev_loss_history = np.load(os.path.join(plots_dir, model_name + "_full_history.npy")).tolist()
    prev_plot_history = np.load(os.path.join(plots_dir, model_name + "_plot_history.npy")).tolist()
else:
    prev_loss_history = None
    prev_plot_history = None

train_iters(encoder, attn_decoder, pairs, input_lang, output_lang, n_iters, random_sampling=True, patience=patience,
            patience_interval=patience_interval, batch_size=batch_size,
            print_every=print_every, plot_every=plot_every, save_every=save_every,
            learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio, max_length=MAX_LENGTH,
            device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
            prev_loss_history=prev_loss_history, prev_plot_history=prev_plot_history)
# evaluate(encoder, attn_decoder, "announcement", input_lang, output_lang, max_length=MAX_LENGTH, device=device)
