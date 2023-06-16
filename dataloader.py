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
        # TODO add left padding for input language
        return self.tokenizer(sentence, add_special_tokens=True, max_length=self.max_length, padding=padding,
                              truncation=truncation).input_ids

    def tokenize_without_truncation(self, sentence):
        return self.tokenizer(sentence, add_special_tokens=True).input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def filter_pairs(pairs, max_length):
    filtered_pairs = [pair for pair in pairs if len(pair[0]) < max_length and len(pair[1]) < max_length]
    return [pair[0] for pair in filtered_pairs], [pair[1] for pair in filtered_pairs]


def prepare_data(path_en, path_de, max_length, training_size, validation_size, loaded_data=-1,
                 data_shuffled_and_filtered=False, device="cpu"):
    input_lang = Lang("en", max_length)
    output_lang = Lang("de", max_length)

    with open(path_en, encoding='utf-8') as f:
        lines_english = [f.readline() for _ in range(loaded_data)]
    with open(path_de, encoding='utf-8') as f:
        lines_german = [f.readline() for _ in range(loaded_data)]

    if data_shuffled_and_filtered:
        validation_en = lines_english[:validation_size]
        train_en = lines_english[validation_size: validation_size + training_size]
        validation_de = lines_german[:validation_size]
        train_de = lines_german[validation_size: validation_size + training_size]

        print("Tokenizing training data...")
        train_en = [torch.LongTensor(input_lang.tokenize(sentence, padding=True)).view(-1, 1).to(device)
                    for sentence in train_en]
        train_de = [torch.LongTensor(output_lang.tokenize(sentence, padding=True)).view(-1, 1).to(device)
                    for sentence in train_de]
        print("Finished tokenizing training data")
    else:
        pairs = [[line_english, line_german]
                 for line_english, line_german in zip(lines_english, lines_german)
                 if len(input_lang.tokenize_without_truncation(line_english)) < max_length and
                 len(output_lang.tokenize_without_truncation(line_german)) < max_length]
        
        print("Filtered to %s sentence pairs" % len(pairs))
        seed(42)
        shuffle(pairs)
        train_pairs = pairs[: training_size]
        validation_pairs = pairs[training_size: training_size + validation_size]
    
        train_en = [pair[0] for pair in train_pairs]
        train_de = [pair[1] for pair in train_pairs]
        print("Train sequences: %s" % len(train_en))

        train_en = [torch.LongTensor(input_lang.tokenize(sentence, padding=True)).view(-1, 1).to(device)
                    for sentence in train_en]
        train_de = [torch.LongTensor(output_lang.tokenize(sentence, padding=True)).view(-1, 1).to(device)
                    for sentence in train_de]
        validation_en = [pair[0] for pair in validation_pairs]
        validation_de = [pair[1] for pair in validation_pairs]

    return input_lang, output_lang, train_en, train_de, validation_en, validation_de
