import random
import time
import torch
import torch.nn as nn
from utils import *
from dataloader import *
from tqdm import tqdm
from plot import show_plot
import os


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

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, pairs, input_lang: Lang, output_lang: Lang, n_iters, print_every=1000,
                plot_every=None, save_every=None, learning_rate=0.01, max_length=None, device="cpu",
                teacher_forcing_ratio=0.5,
                models_dir="model", model_name="model"):
    assert max_length is not None
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(os.path.join(models_dir, model_name)):
        os.mkdir(os.path.join(models_dir, model_name))

    if plot_every is None:
        plot_every = print_every
    if save_every is None:
        save_every = plot_every

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    # tensorFromPair and pairs are defined in loading data
    # training_pairs = [tensors_from_pair(random.choice(pairs), input_lang, output_lang, device=device)
    #                   for _ in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for i in tqdm(range(1, n_iters + 1)):
        training_pair = tensors_from_pair(random.choice(pairs), input_lang, output_lang, device=device)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     max_length=max_length, device=device, teacher_forcing_ratio=teacher_forcing_ratio)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                              iter, iter / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if i % save_every == 0:
            # create model checkpoint
            torch.save(encoder.state_dict(), os.path.join(models_dir, model_name, "encoder_{}.pt".format(i)))
            torch.save(decoder.state_dict(), os.path.join(models_dir, model_name, "decoder_{}.pt".format(i)))

    show_plot(plot_losses, model_name)
    print(plot_losses)
