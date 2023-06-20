"""
This is the main script that ties together all elements of the Seq2Seq model training pipeline.
It initiates the training process, controls the training loop, and handles the data loading and preprocessing.
Also, it calls the functions necessary to evaluate the model's performance and generate plots of the training loss.
"""


import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import train_iters
from dataloader import prepare_data
from io import open
import os
import numpy as np

device: torch.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

model_variant = "gru"  # "gru" or "lstm"
models_dir = "models_" + model_variant
plots_dir = "plots"
resume_training = False

plot_every = 2000
save_every = 2000

initial_validation_size = 20000
start_from_sample = initial_validation_size
n_samples = 500000
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

for i in range(n_hyperparams):
    model_name = hyperparams["model_name"][i]

    teacher_forcing_ratio = hyperparams["teacher_forcing_ratio"][i]
    lr = hyperparams["lr"][i]
    hidden_size = hyperparams["hidden_size"][i]
    n_encoder_layers = hyperparams["n_encoder_layers"][i]
    n_decoder_layers = hyperparams["n_decoder_layers"][i]
    dropout = hyperparams["dropout"][i]
    weight_decay = hyperparams["weight_decay"][i]

    if resume_training:
        encoder = torch.load(os.path.join(models_dir, model_name, "encoder.pt"), map_location=device)
        decoder = torch.load(os.path.join(models_dir, model_name, "decoder.pt"), map_location=device)

        prev_loss_history = np.load(os.path.join(plots_dir, model_name + "_full_history.npy")).tolist()
        prev_plot_history = np.load(os.path.join(plots_dir, model_name + "_plot_history.npy")).tolist()
    else:
        encoder = EncoderRNN(model_variant, input_lang.n_words, hidden_size, num_layers=n_encoder_layers,
                             dropout_p=dropout).to(device)
        decoder = AttnDecoderRNN(model_variant, hidden_size, output_lang.n_words, num_layers=n_decoder_layers,
                                 dropout_p=dropout, max_length=max_length).to(device)
        prev_loss_history = None
        prev_plot_history = None

    train_iters(model_variant, encoder, decoder, english_sequences, german_sequences,
                validation_english, validation_german,
                input_lang, output_lang, n_iters, max_length=max_length, plot_every=plot_every, save_every=save_every,
                learning_rate=lr, weight_decay=weight_decay, teacher_forcing_ratio=teacher_forcing_ratio,
                device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
                prev_loss_history=prev_loss_history, prev_plot_history=prev_plot_history)
