"""
This script is dedicated to final model testing. It loads trained Seq2Seq models and uses the functions defined in
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

initial_testing_size = 40000
max_length = 20
input_lang = Lang("en")
output_lang = Lang("de")

validation_english = open(os.path.join("data", "train.en"), encoding='utf-8').readlines()[-initial_testing_size:]
validation_german = open(os.path.join("data", "train.de"), encoding='utf-8').readlines()[-initial_testing_size:]
zipped = list(zip(validation_english, validation_german))
validation_english = [english for english, german in zipped
                      if len(input_lang.tokenize_without_truncation(english)) < max_length
                      and len(output_lang.tokenize_without_truncation(german)) < max_length]
validation_german = [german for english, german in zipped
                     if len(input_lang.tokenize_without_truncation(english)) < max_length
                     and len(output_lang.tokenize_without_truncation(german)) < max_length]
print(f"testing size: {len(validation_english)}")

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
    model_variant = "gru"
    models_dir = "models_" + model_variant
    model_name = model_names[i]
    plots_dir = "plots"

    encoder = torch.load(os.path.join(models_dir, model_name, "encoder.pt"), map_location=device)
    attn_decoder = torch.load(os.path.join(models_dir, model_name, "decoder.pt"), map_location=device)

    print(f"Evaluating model {model_name}")
    loss = evaluate_loss(model_variant, encoder, attn_decoder, validation_english, validation_german, input_lang,
                         output_lang, max_length, device)
    print(f"Loss: {loss}")
