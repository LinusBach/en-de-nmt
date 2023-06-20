"""
This script is dedicated to model evaluation. It loads trained Seq2Seq models and uses the functions defined in
'evaluate.py' to calculate the performance metrics. It can handle multiple models and provides a systematic way
to compare the performance of different models or the same model at different stages of training.
"""


import torch.backends.mps

from evaluate import evaluate_loss
from dataloader import Lang
from io import open
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'using device: {device}')

initial_validation_size = 5000
max_length = 20
input_lang = Lang("en")
output_lang = Lang("de")

validation_english = open("data/train.en", encoding='utf-8').readlines()[:initial_validation_size]
validation_german = open("data/train.de", encoding='utf-8').readlines()[:initial_validation_size]
zipped = list(zip(validation_english, validation_german))
validation_english = [english for english, german in zipped
                      if len(input_lang.tokenize_without_truncation(english)) < max_length
                      and len(output_lang.tokenize_without_truncation(german)) < max_length]
validation_german = [german for english, german in zipped
                     if len(input_lang.tokenize_without_truncation(english)) < max_length
                     and len(output_lang.tokenize_without_truncation(german)) < max_length]
print(f"validation size: {len(validation_english)}")

n_hyperparams = 8
model_names = [
    "1e-3_lr_256_hidden_4_layers_20p_dropout",
    "1e-4_lr_320_hidden_4_layers_10p_dropout",
    "3e-4_lr_320_hidden_4_layers_20p_dropout",
    "3e-5_lr_320_hidden_5_layers_30p_dropout",
    "50p_tfr_1e-4_lr_320_hidden_4_layers_10p_dropout",
    "80p_tfr_3e-4_lr_320_hidden_5_layers_30p_dropout",
    "100p_tfr_2e-4_lr_400_hidden_8_layers_60p_dropout",
    "100p_tfr_1e-4_lr_320_hidden_6_layers_40p_dropout"
]

for i in range(n_hyperparams):
    models_dir = "models"
    model_name = model_names[i]
    plots_dir = "plots"

    encoder = torch.load(os.path.join(models_dir, model_name, "encoder.pt"), map_location=device)
    attn_decoder = torch.load(os.path.join(models_dir, model_name, "decoder.pt"), map_location=device)
    # prev_loss_history = np.load(os.path.join(plots_dir, model_name + "_full_history.npy")).tolist()
    # prev_plot_history = np.load(os.path.join(plots_dir, model_name + "_plot_history.npy")).tolist()

    print(f"Evaluating model {model_name}")
    loss = evaluate_loss(encoder, attn_decoder, validation_english, validation_german, input_lang, output_lang,
                         max_length, device)
    print(f"Loss: {loss}")
