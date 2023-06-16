from io import open
from transformers import AutoTokenizer
import torch
from random import shuffle, seed
from tqdm import tqdm


class Lang:
    def __init__(self, name, max_length=15):
        self.name = name
        self.max_length = max_length
        if name == "en":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.n_words = 28996
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
            self.n_words = 30000

    def tokenize(self, sentence, padding=True, truncation=True):
        return self.tokenizer(sentence, add_special_tokens=True, max_length=self.max_length, padding=padding,
                              truncation=truncation).input_ids

    def tokenize_without_truncation(self, sentence):
        return self.tokenizer(sentence, add_special_tokens=True).input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def filter_pairs(pairs, max_length):
    filtered_pairs = [pair for pair in pairs if len(pair[0]) < max_length and len(pair[1]) < max_length]
    return [pair[0] for pair in filtered_pairs], [pair[1] for pair in filtered_pairs]


def prepare_data(path_en, path_de, max_length, training_size, validation_size, device="cpu"):
    input_lang = Lang("en", max_length)
    output_lang = Lang("de", max_length)

    lines_english = open(path_en, encoding='utf-8').readlines()
    lines_german = open(path_de, encoding='utf-8').readlines()
    pairs = [[line_english, line_german]
             for line_english, line_german in zip(lines_english, lines_german)
             if len(input_lang.tokenize_without_truncation(line_english)) < max_length and
             len(output_lang.tokenize_without_truncation(line_german)) < max_length]
    print("Filtered to %s sentence pairs" % len(pairs))
    seed(42)
    shuffle(pairs)
    train_pairs = pairs[: training_size]
    validation_pairs = pairs[training_size: training_size + validation_size]

    train_en_sequences = [pair[0] for pair in train_pairs]
    train_de_sequences = [pair[1] for pair in train_pairs]
    print("Train sequences: %s" % len(train_en_sequences))

    train_en_sequences = [torch.LongTensor(input_lang.tokenize(train_en_sequences[i], padding=True)).view(-1, 1).to(device)
                          for i in tqdm(range(len(train_en_sequences)))]
    train_de_sequences = [torch.LongTensor(output_lang.tokenize(sequence, padding=True)).view(-1, 1).to(device)
                          for sequence in train_de_sequences]
    validation_en_sequences = [pair[0] for pair in validation_pairs]
    validation_de_sequences = [pair[1] for pair in validation_pairs]

    return input_lang, output_lang, train_en_sequences, train_de_sequences, \
        validation_en_sequences, validation_de_sequences
