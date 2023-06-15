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

initial_validation_size = 15000
start_from_sample = initial_validation_size
shuffling = False
n_samples = 400000
max_length = 20  # max length of 30 retains around 1/3 of the data; 20 => 1/8
n_iters = 40000

input_lang, output_lang, english_sequences, german_sequences = prepare_data('data/train.en', 'data/train.de',
                                                                            max_length, n_samples,
                                                                            start_from_sample, device=device)

validation_english = open("data/train.en", encoding='utf-8').readlines()[:initial_validation_size]
validation_german = open("data/train.de", encoding='utf-8').readlines()[:initial_validation_size]
zipped = list(zip(validation_english, validation_german))
validation_english = [english for english, german in zipped
                      if len(input_lang.tokenize_without_truncation(english)) < max_length
                      and len(output_lang.tokenize_without_truncation(german)) < max_length]
validation_german = [german for english, german in zipped
                     if len(input_lang.tokenize_without_truncation(english)) < max_length
                     and len(output_lang.tokenize_without_truncation(german)) < max_length]
print(f'validation size: {len(validation_english)}')

print_every = 1000
plot_every = 2000
save_every = 2000

patience = 1000  # early stopping
patience_interval = 1000
# batch_first = True
# batch_size = 1

n_hyperparams = 6
hyperparams = {"model_name": ["100p_tfr_5e-5_lr_512_hidden_8_layers_60p_dropout_1e-4_weight_decay",
                              "100p_tfr_1e-4_lr_512_hidden_8_layers_50p_dropout",
                              "50p_tfr_1e-4_lr_320_hidden_4_layers_10p_dropout",
                              "80p_tfr_3e-4_lr_320_hidden_5_layers_30p_dropout",
                              "100p_tfr_2e-4_lr_400_hidden_8_layers_60p_dropout",
                              "100p_tfr_1e-4_lr_320_hidden_6_layers_40p_dropout"],
               "weight_decay": [1e-4, 0, 0, 0, 0, 0],
               "teacher_forcing_ratio": [1, 1, 0.5, 0.8, 1, 1],
               "lr": [5e-5, 1e-4, 1e-4, 3e-4, 2e-4, 1e-4],
               "hidden_size": [512, 512, 320, 320, 400, 320],
               "n_encoder_layers": [8, 8, 4, 5, 8, 6],
               "n_decoder_layers": [8, 8, 4, 5, 8, 6],
               "dropout": [0.6, 0.5, 0.1, 0.3, 0.6, 0.4]}

for i in range(2, n_hyperparams):
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
    weight_decay = hyperparams["weight_decay"][i]

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
                learning_rate=lr, weight_decay=weight_decay, teacher_forcing_ratio=teacher_forcing_ratio,
                device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
                prev_loss_history=prev_loss_history, prev_plot_history=prev_plot_history)
