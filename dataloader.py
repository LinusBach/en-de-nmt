from io import open
import unicodedata
import re
from transformers import AutoTokenizer

import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 15


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

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence, add_special_tokens=True, max_length=self.max_length, truncation=True).input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def filter_pairs(pairs):
    filtered_pairs = [pair for pair in pairs if len(pair[0]) < MAX_LENGTH and len(pair[1]) < MAX_LENGTH]
    return [pair[0] for pair in filtered_pairs], [pair[1] for pair in filtered_pairs]


def prepare_data(path_en, path_de, max_length, sample_size=-1, start_from_sample=0, device="cpu"):
    input_lang = Lang("en", max_length)
    output_lang = Lang("de", max_length)

    lines1 = open(path_en, encoding='utf-8').readlines()[start_from_sample:sample_size + start_from_sample]
    lines2 = open(path_de, encoding='utf-8').readlines()[start_from_sample:sample_size + start_from_sample]
    pairs = [[input_lang.tokenize(line1), output_lang.tokenize(line2)] for line1, line2 in zip(lines1, lines2)]

    print("Read %s sentence pairs" % len(pairs))
    en_pairs, de_pairs = filter_pairs(pairs)
    en_pairs = torch.tensor(en_pairs, dtype=torch.long, device=device)
    de_pairs = torch.tensor(de_pairs, dtype=torch.long, device=device)
    print("Trimmed to %s sentence pairs based on length of tokenization" % len(pairs))

    return input_lang, output_lang, en_pairs, de_pairs
