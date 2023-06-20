"""
This script contains functions for evaluating the performance of the Seq2Seq model.
It calculates the Cross-Entropy loss and the BLEU score on a given dataset.
It also provides an inference function that uses the trained model to translate an input sentence.
"""


import torch
from dataloader import Lang, de_CLS_token, de_SEP_token
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from typing import List, Tuple


def evaluate_loss(model_variant: str, encoder: EncoderRNN, decoder: AttnDecoderRNN, input_sentences: List[str],
                  target_sentences: List[str], input_lang: Lang, output_lang: Lang,
                  max_length: int, device: torch.device) -> float:
    encoder.eval()
    decoder.eval()

    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(len(input_sentences)):
        input_tensor = torch.LongTensor(input_lang.tokenize(input_sentences[i])).view(-1, 1).to(device)
        target_tensor = torch.LongTensor(output_lang.tokenize(target_sentences[i])).view(-1, 1).to(device)

        encoder_hidden = encoder.init_hidden(device=device)

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss: torch.Tensor = torch.tensor(0.0, device=device)

        for j in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[j], encoder_hidden)
            encoder_outputs[j] = encoder_output[0, 0]

        decoder_input = torch.tensor([[de_CLS_token]], device=device)
        decoder_hidden = encoder_hidden

        for j in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            top_vector, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[j])
            if decoder_input.item() == de_SEP_token:
                break

        loss /= target_length
        total_loss += loss.item()

    return total_loss / len(input_sentences)


def evaluate_bleu(model_variant: str, encoder: EncoderRNN, decoder: AttnDecoderRNN, input_sentences: List[str],
                  reference_sentences: List[str], input_lang: Lang, output_lang: Lang, max_length: int,
                  device: torch.device) -> float:
    bleu_scores = []

    for i in range(len(input_sentences)):
        prediction, _ = inference(model_variant, encoder, decoder, input_sentences[i], input_lang, output_lang,
                                  max_length, device)
        bleu_score = sentence_bleu(reference_sentences[i], prediction)
        bleu_scores.append(bleu_score)

    return float(np.mean(bleu_scores))


def inference(model_variant: str, encoder: EncoderRNN, decoder: AttnDecoderRNN, sentence: str,
              input_lang: Lang, output_lang: Lang,
              max_length: int, device: torch.device) -> Tuple[List[str], torch.Tensor]:
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = torch.LongTensor(input_lang.tokenize(sentence)).view(-1, 1).to(device)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden(device=device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],
                                                     encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[de_CLS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[i] = decoder_attention.data
            top_vector, top_index = decoder_output.data.topk(1)
            if top_index.item() == de_SEP_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.decode(top_index.item()))

            decoder_input = top_index.squeeze().detach()

        return decoded_words, decoder_attentions[:i + 1]
