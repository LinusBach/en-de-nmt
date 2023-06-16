import random
import torch.nn as nn
from utils import *
from dataloader import *
from tqdm import tqdm
from plot import plot
import os
import numpy as np
from evaluation_functions import evaluate_loss, evaluate_bleu, evaluate_meteor, evaluate_bertscore


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=None, device="cpu", teacher_forcing_ratio=0.5):
    assert max_length is not None

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
            # print(decoder_output.shape, target_tensor[i].shape, target_tensor[i])
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == de_SEP_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, input_sequences, output_sequences,
                validation_input_sentences, validation_output_sentences, input_lang, output_lang, n_iters, max_length,
                shuffling=False, batch_size=1, evaluation_model="microsoft/deberta-v2-xlarge-mnli",
                patience=100, patience_interval=20, print_every=1000, plot_every=None, save_every=None,
                learning_rate=0.01, weight_decay=1e-5, teacher_forcing_ratio=0.5,
                device="cpu", models_dir="models", model_name="model", plots_dir="plots",
                prev_loss_history=None, prev_plot_history=None):
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(os.path.join(models_dir, model_name)):
        os.mkdir(os.path.join(models_dir, model_name))
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    if not os.path.exists(os.path.join(plots_dir, model_name)):
        os.mkdir(os.path.join(plots_dir, model_name))

    if plot_every is None:
        plot_every = print_every
    if save_every is None:
        save_every = plot_every

    losses = [] if prev_loss_history is None else prev_loss_history
    plot_losses = [] if prev_plot_history is None else prev_plot_history
    eval_losses = []
    bertscores = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    patience_loss_total = 0  # Reset every patience_interval
    min_interval_loss = np.inf
    steps_without_improvement = 0

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
        print_loss_total += loss
        plot_loss_total += loss
        patience_loss_total += loss
        losses.append(loss)

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                              iter, iter / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            eval_loss = evaluate_loss(encoder, decoder, validation_input_sentences, validation_output_sentences,
                                      input_lang, output_lang, max_length=max_length, device=device)
            eval_losses.append(eval_loss)
            bertscore = evaluate_bertscore(encoder, decoder, validation_input_sentences, validation_output_sentences,
                                           input_lang, output_lang, max_length=max_length,
                                           evaluation_model=evaluation_model, device=device)
            bertscores.append(bertscore)
            # print("Evaluation loss: ", eval_loss)
            plot(eval_losses, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_validation_loss")
            plot(bertscores, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_bertscore")
            encoder.train()
            decoder.train()

            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot(plot_losses, plot_every, plots_dir=plots_dir, model_name=model_name, suffix="_training_loss")
            np.save(os.path.join(plots_dir, model_name + "_plot_history.npy"), np.array(plot_losses))
            np.save(os.path.join(plots_dir, model_name + "_eval_history.npy"), np.array(eval_losses))

        if i % save_every == 0:
            # create models checkpoint
            encoder.to("cpu")
            decoder.to("cpu")
            torch.save(encoder, os.path.join(models_dir, model_name, "encoder.pt"))
            torch.save(decoder, os.path.join(models_dir, model_name, "decoder.pt"))
            encoder.to(device)
            decoder.to(device)
            # save loss history
            np.save(os.path.join(plots_dir, model_name + "_full_history.npy"), np.array(losses))

        if i % patience_interval == 0:
            patience_loss_avg = patience_loss_total / patience_interval
            patience_loss_total = 0
            if patience_loss_avg < min_interval_loss:
                min_interval_loss = patience_loss_avg
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    # create models checkpoint
                    encoder.to("cpu")
                    decoder.to("cpu")
                    torch.save(encoder, os.path.join(models_dir, model_name, "encoder.pt"))
                    torch.save(decoder, os.path.join(models_dir, model_name, "decoder.pt"))
                    # save loss history
                    np.save(os.path.join(plots_dir, model_name + "_full_history.npy"), np.array(losses))
                    print("Stopping early")
                    break
