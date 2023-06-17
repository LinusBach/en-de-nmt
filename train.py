import random

import torch
import torch.nn as nn
from utils import *
from dataloader import *
from tqdm import tqdm
from plot import plot
import os
import numpy as np
from evaluation_functions import evaluate_loss, evaluate_bleu, evaluate_meteor, evaluate_bertscore


def train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          batch_size=1, max_length=None, device="cpu", teacher_forcing_ratio=0.5):
    assert max_length is not None

    encoder_hidden = encoder.init_hidden(device=device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensors.size(-2)
    target_length = target_tensors.size(-1)
    # print(input_tensors.shape, target_tensors.shape)

    encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

    loss: torch.Tensor = torch.tensor(0.0, device=device)

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensors[:, i], encoder_hidden)
        # print(encoder_output.shape, encoder_outputs[:, i].shape)
        encoder_outputs[:, i] = encoder_output[0]

    decoder_inputs = torch.LongTensor([de_CLS_token] * batch_size).view(-1, 1).to(device)
    decoder_hidden = encoder_hidden
    # print("encoder_hidden.shape", encoder_hidden.shape)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_outputs, decoder_hidden, decoder_attention = decoder(
                decoder_inputs, decoder_hidden, encoder_outputs)
            # print(decoder_outputs.shape, target_tensor[i].shape, target_tensor[i])
            # print(decoder_outputs.shape, target_tensors[:, i].shape)
            # print(torch.sum(decoder_outputs, dim=1), target_tensors[:, i])
            loss += criterion(decoder_outputs, target_tensors[:, i])
            # print("loss:", loss)
            decoder_inputs = target_tensors[:, i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_outputs, decoder_hidden, decoder_attention = decoder(
                decoder_inputs, decoder_hidden, encoder_outputs)
            topv, topi = decoder_outputs.topk(1)
            decoder_inputs = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_outputs, target_tensors[:, i])
            if decoder_inputs.item() == de_SEP_token:  # TODO
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, input_sequences, output_sequences,
                validation_input_sentences, validation_output_sentences, input_lang, output_lang, epochs, max_length,
                shuffling=False, batch_size=1, evaluation_model="microsoft/deberta-v2-xlarge-mnli",
                patience=10, patience_interval=20, print_every=1000, plot_every=None, save_every=None,
                learning_rate=0.01, weight_decay=1e-5, teacher_forcing_ratio=0.5,
                device="cpu", models_dir="models", model_name="model", plots_dir="plots",
                prev_loss_history=None, prev_training_loss_plot_history=None, prev_eval_loss_plot_history=None,
                prev_bertscore_plot_history=None):
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(os.path.join(models_dir, model_name)):
        os.mkdir(os.path.join(models_dir, model_name))
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    if not os.path.exists(os.path.join(plots_dir, model_name)):
        os.mkdir(os.path.join(plots_dir, model_name))

    losses_list = [] if prev_loss_history is None else prev_loss_history
    plot_losses = [] if prev_training_loss_plot_history is None else prev_training_loss_plot_history
    eval_losses = [] if prev_eval_loss_plot_history is None else prev_eval_loss_plot_history
    bertscores = [] if prev_bertscore_plot_history is None else prev_bertscore_plot_history
    min_epoch_loss = np.inf
    epochs_without_improvement = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        print("epoch:", epoch)
        encoder.train()
        decoder.train()

        epoch_loss = 0
        losses = []
        for j in tqdm(range(input_sequences.size(0) // batch_size)):
            input_tensors = input_sequences[j * batch_size: (j + 1) * batch_size]
            # print("input_tensors.shape", input_tensors.shape)
            target_tensors = output_sequences[j * batch_size: (j + 1) * batch_size]

            loss = train(input_tensors, target_tensors, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size,
                         max_length=max_length, device=device, teacher_forcing_ratio=teacher_forcing_ratio)
            epoch_loss += loss
            losses.append(loss)

        epoch_loss /= (input_sequences.size(0) // batch_size)
        plot_losses.append(epoch_loss)
        losses_list.append(losses)
        print(losses)

        eval_loss = evaluate_loss(encoder, decoder, validation_input_sentences, validation_output_sentences,
                                  input_lang, output_lang, max_length=max_length, device=device)
        eval_losses.append(eval_loss)

        bertscore = evaluate_bertscore(encoder, decoder, validation_input_sentences, validation_output_sentences,
                                       input_lang, output_lang, max_length=max_length,
                                       evaluation_model=evaluation_model, device=device)
        bertscores.append(bertscore)

        print("epoch_loss:", epoch_loss)
        print("eval_loss:", eval_loss)
        print("bertscore:", bertscore)

        plot(eval_losses, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_validation_loss")
        plot(bertscores, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_bertscore")
        plot(plot_losses, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_training_loss")


        np.save(os.path.join(plots_dir, model_name, "training_loss_plot_history.npy"), np.array(plot_losses))
        np.save(os.path.join(plots_dir, model_name, "eval_loss_plot_history.npy"), np.array(eval_losses))
        np.save(os.path.join(plots_dir, model_name, "bertscore_plot_history.npy"), np.array(eval_losses))

        # create models checkpoint
        # encoder.to("cpu")
        # decoder.to("cpu")
        torch.save(encoder, os.path.join(models_dir, model_name, "encoder.pt"))
        torch.save(decoder, os.path.join(models_dir, model_name, "decoder.pt"))
        # encoder.to(device)
        # decoder.to(device)
        # save loss history
        np.save(os.path.join(plots_dir, model_name, "full_training_loss_history.npy"), np.array(losses_list))

        if epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Stopping early")
                break
