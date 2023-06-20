import torch
from utils import *
from dataloader import Lang
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
from tqdm import tqdm
from time import time
from evaluate import load
bertscore = load("bertscore")


def evaluate_loss(encoder, decoder, input_sentences, target_sentences, input_lang: Lang, output_lang: Lang, max_length,
                  device, batch_size=128):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    with torch.no_grad():
        criterion = torch.nn.CrossEntropyLoss()

        for i in tqdm(range(len(input_sentences) // batch_size)):
            input_tensors = torch.LongTensor(input_lang.tokenize(input_sentences[i * batch_size: (i + 1) * batch_size]))\
                .view(batch_size, -1, 1).to(device)
            target_tensors = torch.LongTensor(output_lang.tokenize(target_sentences[i * batch_size: (i + 1) * batch_size]))\
                .view(batch_size, -1).to(device)
            # print(input_tensors.shape, target_tensors.shape)

            encoder_hidden = encoder.init_hidden(batch_size=batch_size, device=device)

            input_length = input_tensors.size(-2)
            target_length = target_tensors.size(-1)

            encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

            loss: torch.Tensor = torch.tensor(0.0, device=device)

            for j in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensors[:, j], encoder_hidden, batch_size=batch_size)
                encoder_outputs[:, j] = encoder_output[0]

            decoder_inputs = torch.LongTensor([de_CLS_token] * batch_size).view(-1, 1).to(device)
            decoder_hidden = encoder_hidden

            for j in range(target_length):
                decoder_outputs, decoder_hidden, decoder_attention = decoder(
                    decoder_inputs, decoder_hidden, encoder_outputs, batch_size=batch_size)
                topv, topi = decoder_outputs.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_outputs, target_tensors[:, j])
                # if decoder_input.item() == de_SEP_token:
                #     break

            loss /= target_length
            total_loss += loss.item()

    return total_loss / len(input_sentences) // batch_size


def evaluate_bleu(encoder, decoder, input_sentences, reference_sentences, input_lang: Lang, output_lang: Lang, max_length,
                  device):
    bleu_scores = []

    for i in range(len(input_sentences)):
        prediction, _ = inference(encoder, decoder, input_sentences[i], input_lang, output_lang, max_length, device)
        # print("evaluating: ", input_sentences[i], " -> ", prediction, " vs ", reference_sentences[i])
        bleu_score = sentence_bleu(reference_sentences[i], prediction)
        bleu_scores.append(bleu_score)

    # print("BLEU score: ", np.mean(bleu_scores))
    return np.mean(bleu_scores)


def evaluate_meteor(encoder, decoder, input_sentences, reference_sentences, input_lang: Lang, output_lang: Lang,
                    max_length, device):
    meteor_scores = []

    for i in range(len(input_sentences)):
        prediction = inference(encoder, decoder, input_sentences[i], input_lang, output_lang, max_length, device)
        meteor_score = single_meteor_score(reference_sentences[i], prediction)
        meteor_scores.append(meteor_score)

    return np.mean(meteor_scores)


def evaluate_bertscore(encoder, decoder, input_sentences, reference_sentences, input_lang: Lang, output_lang: Lang,
                       max_length, evaluation_model="facebook/bart-large-mnli", device="cpu", batch_size=128):

    bert_scores = []
    for i in range(len(input_sentences) // batch_size):
        predictions = inference(encoder, decoder, input_sentences[i * batch_size: (i + 1) * batch_size],
                                input_lang, output_lang, max_length, device,
                                batch_size=batch_size)
        bert_scores.append(bertscore.compute(predictions=predictions,
                                             references=reference_sentences[i * batch_size: (i + 1) * batch_size],
                           model_type=evaluation_model, lang="de", device=device)["f1"])

    return np.average(bert_scores)


def inference(encoder, decoder, sentences, input_lang: Lang, output_lang: Lang, max_length, device, batch_size=1):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # input_tensors = tensor_from_sentence(input_lang, sentence, device=device)
        input_tensors = torch.LongTensor(input_lang.tokenize(sentences)).view(batch_size, -1, 1).to(device)
        input_length = input_tensors.size(-2)
        encoder_hidden = encoder.init_hidden(batch_size=batch_size, device=device)

        encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensors[:, i], encoder_hidden, batch_size=batch_size)
            encoder_outputs[i] += encoder_output[0, 0]

        # decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_inputs = torch.LongTensor([de_CLS_token] * batch_size).view(-1, 1).to(device)
        decoder_hidden = encoder_hidden

        decoded_tokens_list = [[de_CLS_token] for _ in range(batch_size)]

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_inputs, decoder_hidden, encoder_outputs, batch_size=batch_size)
            topv, topi = decoder_output.data.topk(1)
            # print(topi.shape)
            for j in range(batch_size):
                decoded_tokens_list[j].append(topi[j].item())
            # decoded_tokens.append([topi[i].item() for i in range(max_length)])
            # if topi.item() == de_SEP_token:
            #     break

            decoder_inputs = topi.squeeze().detach()

        decoded_sentences = [output_lang.decode(decoded_tokens) for decoded_tokens in decoded_tokens_list]

        return decoded_sentences
