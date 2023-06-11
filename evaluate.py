import torch
from utils import *
from dataloader import Lang
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm


def evaluate(encoder, decoder, input_sentences, reference_sentences, input_lang: Lang, output_lang: Lang, max_length,
             device):
    bleu_scores = []

    for i in range(len(input_sentences)):
        prediction, _ = inference(encoder, decoder, input_sentences[i], input_lang, output_lang, max_length, device)
        # print("evaluating: ", input_sentences[i], " -> ", prediction, " vs ", reference_sentences[i])
        bleu_score = sentence_bleu(reference_sentences[i], prediction)
        bleu_scores.append(bleu_score)

    # print("BLEU score: ", np.mean(bleu_scores))
    return np.mean(bleu_scores)


def inference(encoder, decoder, sentence, input_lang: Lang, output_lang: Lang, max_length, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # input_tensor = tensor_from_sentence(input_lang, sentence, device=device)
        input_tensor = torch.LongTensor(input_lang.tokenize(sentence)).view(-1, 1).to(device)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden(device=device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],
                                                     encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        # decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_input = torch.tensor([[de_CLS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[i] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == de_SEP_token:
                decoded_words.append('<EOS>')
                break
            else:
                # print(topi.item())
                decoded_words.append(output_lang.decode(topi.item()))

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:i + 1]
