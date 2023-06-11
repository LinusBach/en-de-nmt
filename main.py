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
model_name = "new_tokenizer"
plots_dir = "plots"
resume_training = True

n_iters = 1000
start_from_sample = 0
shuffling = False
n_samples = 10000
max_length = 30

patience = 20  # early stopping
patience_interval = 100
# batch_first = True
# batch_size = 1

teacher_forcing_ratio = 1
lr = 1e-4
hidden_size = 512
n_encoder_layers = 4
n_decoder_layers = 4
dropout = 0.1

print_every = 100
plot_every = 100
save_every = 1000

input_lang, output_lang, english_sequences, german_sequences = prepare_data('data/train.en', 'data/train.de',
                                                                            max_length, n_samples,
                                                                            start_from_sample, device=device)
print(f'English sentences: {len(english_sequences)}')
print(f'German sentences: {len(german_sequences)}')

if resume_training:
    encoder = torch.load(os.path.join(models_dir, model_name, "encoder.pt"), map_location=device)
    # encoder.flatten_parameters()
    attn_decoder = torch.load(os.path.join(models_dir, model_name, "decoder.pt"), map_location=device)
    # attn_decoder.flatten_parameters()
    prev_loss_history = np.load(os.path.join(plots_dir, model_name + "_full_history.npy")).tolist()
    prev_plot_history = np.load(os.path.join(plots_dir, model_name + "_plot_history.npy")).tolist()
else:
    encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers=n_encoder_layers, dropout_p=dropout).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers=n_decoder_layers,
                                  dropout_p=dropout, max_length=max_length).to(device)
    prev_loss_history = None
    prev_plot_history = None

train_iters(encoder, attn_decoder, english_sequences, german_sequences, n_iters, max_length=max_length,
            shuffling=shuffling, patience=patience, patience_interval=patience_interval,  # batch_size=batch_size,
            print_every=print_every, plot_every=plot_every, save_every=save_every,
            learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio,
            device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
            prev_loss_history=prev_loss_history, prev_plot_history=prev_plot_history)
# evaluate(encoder, attn_decoder, "announcement", input_lang, output_lang, max_length=MAX_LENGTH, device=device)
