import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import train_iters
from dataloader import prepare_data
from io import open
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

n_iters = 40000
start_from_sample = 0
shuffling = False
n_samples = 500000
max_length = 20  # max length of 30 retains around 1/3 of the data

input_lang, output_lang, english_sequences, german_sequences = prepare_data('data/train.en', 'data/train.de',
                                                                            max_length, n_samples,
                                                                            start_from_sample, device=device)

validation_size = 1000
validation_english = open("data/newstest2015.en", encoding='utf-8').readlines()[:validation_size]
validation_english = [line for line in validation_english if len(input_lang.tokenize(line)) < max_length]
validation_german = open("data/newstest2015.de", encoding='utf-8').readlines()[:validation_size]
validation_german = [line for line in validation_german if len(output_lang.tokenize(line)) < max_length]

print_every = 100
plot_every = 1000
save_every = 1000

patience = 10  # early stopping
patience_interval = 1000
# batch_first = True
# batch_size = 1

n_hyperparams = 4
hyperparams = {"model_name": ["1e-4_lr_320_hidden_4_layers_10p_dropout",
                              "3e-5_lr_320_hidden_5_layers_30p_dropout",
                              "3e-4_lr_320_hidden_4_layers_20p_dropout",
                              "1e-3_lr_256_hidden_4_layers_20p_dropout"],
               "teacher_forcing_ratio": [1, 1, 1, 1],
               "lr": [1e-4, 3e-5, 3e-4, 1e-3],
               "hidden_size": [320, 320, 320, 256],
               "n_encoder_layers": [4, 5, 4, 3],
               "n_decoder_layers": [4, 5, 4, 3],
               "dropout": [0.1, 0.3, 0.2, 0.2]}

for i in range(n_hyperparams):
    models_dir = "models"
    model_name = hyperparams["model_name"][i]
    plots_dir = "plots"
    resume_training = False

    teacher_forcing_ratio = hyperparams["teacher_forcing_ratio"][i]
    lr = hyperparams["lr"][i]
    hidden_size = hyperparams["hidden_size"][i]
    n_encoder_layers = hyperparams["n_encoder_layers"][i]
    n_decoder_layers = hyperparams["n_decoder_layers"][i]
    dropout = hyperparams["dropout"][i]

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

    train_iters(encoder, attn_decoder, english_sequences, german_sequences, validation_english, validation_german,
                input_lang, output_lang, n_iters, max_length=max_length,
                shuffling=shuffling, patience=patience, patience_interval=patience_interval,  # batch_size=batch_size,
                print_every=print_every, plot_every=plot_every, save_every=save_every,
                learning_rate=lr, teacher_forcing_ratio=teacher_forcing_ratio,
                device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
                prev_loss_history=prev_loss_history, prev_plot_history=prev_plot_history)
