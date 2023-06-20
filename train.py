"""
This script is used to train the Seq2Seq model. It loads the preprocessed data, initializes the model,
and trains the model using the specified hyperparameters.
The model parameters are saved after each training epoch.
Also, a plot of the training loss is created after each epoch using the functions provided in plot.py.
"""


import random
import torch.nn as nn
from tqdm import tqdm
from plot import plot
import os
import torch
import numpy as np
from evaluate import evaluate_loss
from dataloader import Lang, de_CLS_token, de_SEP_token
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from typing import List


def train(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: EncoderRNN, decoder: AttnDecoderRNN,
          encoder_optimizer: torch.optim.Optimizer, decoder_optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, max_length: int = None, device: torch.device = "cpu",
          teacher_forcing_ratio: float = 0.5) -> float:
    encoder_hidden = encoder.init_hidden(device=device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss: torch.Tensor = torch.tensor(0.0, device=device)

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[de_CLS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            top_vector, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == de_SEP_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder: EncoderRNN, decoder: AttnDecoderRNN,
                input_sequences: List[torch.Tensor], output_sequences: List[torch.Tensor],
                validation_input_sentences: List[str], validation_output_sentences: List[str],
                input_lang: Lang, output_lang: Lang,
                n_iters: int, max_length: int,
                plot_every: int = 1000, save_every: int = 1000,
                learning_rate: float = 0.01, weight_decay: float = 1e-5, teacher_forcing_ratio: float = 0.5,
                device: torch.device = "cpu", models_dir: str = "models", model_name: str = "model",
                plots_dir: str = "plots", prev_loss_history: List[float] = None, prev_plot_history: List[float] = None):
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(os.path.join(models_dir, model_name)):
        os.mkdir(os.path.join(models_dir, model_name))
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    losses = [] if prev_loss_history is None else prev_loss_history
    plot_losses = [] if prev_plot_history is None else prev_plot_history
    eval_losses = []
    plot_loss_total = 0  # Reset every plot_every
    patience_loss_total = 0  # Reset every patience_interval
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    encoder.train()
    decoder.train()

    for i in tqdm(range(1, n_iters + 1)):
        input_tensor = input_sequences[i - 1]
        target_tensor = output_sequences[i - 1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     max_length=max_length, device=device, teacher_forcing_ratio=teacher_forcing_ratio)
        plot_loss_total += loss
        patience_loss_total += loss
        losses.append(loss)

        if i % plot_every == 0:
            eval_loss = evaluate_loss(encoder, decoder, validation_input_sentences, validation_output_sentences,
                                      input_lang, output_lang, max_length=max_length, device=device)
            eval_losses.append(eval_loss)
            plot(eval_losses, plot_every, plots_dir=plots_dir, model_name=model_name + "_validation")
            encoder.train()
            decoder.train()

            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot(plot_losses, plot_every, plots_dir=plots_dir, model_name=model_name)
            np.save(os.path.join(plots_dir, model_name + "_plot_history.npy"), np.array(plot_losses))
            np.save(os.path.join(plots_dir, model_name + "_eval_history.npy"), np.array(eval_losses))
            # save loss history
            np.save(os.path.join(plots_dir, model_name + "_full_history.npy"), np.array(losses))

        if i % save_every == 0:
            # create models checkpoint
            encoder.to("cpu")
            decoder.to("cpu")
            torch.save(encoder, os.path.join(models_dir, model_name, "encoder.pt"))
            torch.save(decoder, os.path.join(models_dir, model_name, "decoder.pt"))
            encoder.to(device)
            decoder.to(device)
