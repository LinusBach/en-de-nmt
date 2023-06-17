import torch.backends.mps

from decoder import AttnDecoderRNN
from encoder import EncoderRNN
from train import train_iters
from dataloader import prepare_data
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

models_dir = "models_gru"
plots_dir = "plots"
resume_training = False

epochs = 10
train_size = 2000
validation_size = 200
evaluation_model = "facebook/bart-large-mnli"
max_length = 30  # max length of 40 retains a little over 1/2 of the data; 30 retains around 1/3; 20 => 1/8
data_shuffled_and_filtered = True
shuffling = False
if data_shuffled_and_filtered:
    datafile_en = "data/train_st_" + str(max_length) + ".en"
    datafile_de = "data/train_st_" + str(max_length) + ".de"
    data_to_load = train_size + validation_size
else:
    datafile_en = "data/train.en"
    datafile_de = "data/train.de"
    data_to_load = 1000000

input_lang, output_lang, english_sequences, german_sequences, validation_english, validation_german = \
    prepare_data(datafile_en, datafile_de, max_length, train_size, validation_size,
                 loaded_data=data_to_load, data_shuffled_and_filtered=data_shuffled_and_filtered, device=device)

patience = 1000  # early stopping
patience_interval = 1000
batch_first = False
batch_size = 128

print_every = 1000
plot_every = 1000
save_every = 1000

n_hyperparams = 1
hyperparams = {"model_name": [
    "1e-5_lr_1_layer_100_hidden_more_regularization",
    "5e-5_lr_1_layer_100_hidden_2",
    "5e-5_lr_1_layer_100_hidden",
    "100p_tfr_5e-5_lr_512_hidden_8_layers_60p_dropout_1e-4_weight_decay",
    "100p_tfr_1e-4_lr_512_hidden_8_layers_50p_dropout",
    "50p_tfr_1e-4_lr_320_hidden_4_layers_10p_dropout",
    "80p_tfr_3e-4_lr_320_hidden_5_layers_30p_dropout",
    "100p_tfr_2e-4_lr_400_hidden_8_layers_60p_dropout",
    "100p_tfr_1e-4_lr_320_hidden_6_layers_40p_dropout"],
               "weight_decay": [1e-4, 1e-4, 1e-4, 1e-4, 0, 0, 0, 0, 0],
               "teacher_forcing_ratio": [1, 1, 1, 1, 1, 0.5, 0.8, 1, 1],
               "lr": [1e-3, 5e-5, 5e-5, 5e-5, 1e-4, 1e-4, 3e-4, 2e-4, 1e-4],
               "hidden_size": [100, 100, 100, 512, 512, 320, 320, 400, 320],
               "n_encoder_layers": [2, 1, 1, 8, 8, 4, 5, 8, 6],
               "n_decoder_layers": [2, 1, 1, 8, 8, 4, 5, 8, 6],
               "dropout": [0.3, 0.3, 0.3, 0.6, 0.5, 0.1, 0.3, 0.6, 0.4]}

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
        prev_loss_history = np.load(os.path.join(plots_dir, model_name, "full_history.npy")).tolist()
        prev_training_loss_plot_history = np.load(os.path.join(plots_dir, model_name,
                                                               "training_loss_plot_history.npy")).tolist()
        prev_eval_loss_plot_history = np.load(os.path.join(plots_dir, model_name,
                                                           "eval_loss_plot_history.npy")).tolist()
        prev_bertscore_plot_history = np.load(os.path.join(plots_dir, model_name,
                                                           "bertscore_plot_history.npy")).tolist()
    else:
        encoder = EncoderRNN(input_lang.n_words, hidden_size, num_layers=n_encoder_layers, dropout_p=dropout,
                             batch_size=batch_size, batch_first=batch_first).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, num_layers=n_decoder_layers,
                                 dropout_p=dropout, max_length=max_length,
                                 batch_size=batch_size, batch_first=batch_first).to(device)
        prev_loss_history = None
        prev_training_loss_plot_history = None
        prev_eval_loss_plot_history = None
        prev_bertscore_plot_history = None

    train_iters(encoder, decoder, english_sequences, german_sequences, validation_english, validation_german,
                input_lang, output_lang, epochs, max_length=max_length, patience=patience,
                patience_interval=patience_interval,  batch_size=batch_size, evaluation_model=evaluation_model,
                print_every=print_every, plot_every=plot_every, save_every=save_every,
                learning_rate=lr, weight_decay=weight_decay, teacher_forcing_ratio=teacher_forcing_ratio,
                device=device, models_dir=models_dir, model_name=model_name, plots_dir=plots_dir,
                prev_loss_history=prev_loss_history, prev_training_loss_plot_history=prev_training_loss_plot_history,
                prev_eval_loss_plot_history=prev_eval_loss_plot_history,
                prev_bertscore_plot_history=prev_bertscore_plot_history)
