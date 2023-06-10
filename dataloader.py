from io import open
import unicodedata
import re
from transformers import AutoTokenizer

import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30


class Lang:
    def __init__(self, name):
        self.name = name
        # if name == "en":
        #     self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained("bert-base-german-uncased")
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        with open("data/vocab." + name, encoding='utf-8') as f:
            for line in f:
                self.n_words += 1
                word = line.strip()
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
        # self.n_words = 50002  # Count SOS and EOS


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?\-#]+", r" ", s)
    return s


def read_langs(path_en, path_de, sample_size):
    print("Reading lines...")

    # Read the file and split into lines
    # lines1 = open(path_en, encoding='utf-8').\
    #     read().strip().split('\n')
    lines1 = open(path_en, encoding='utf-8').readlines()[:sample_size]

    # lines2 = open(path_de, encoding='utf-8').\
    #     read().strip().split('\n')
    lines2 = open(path_de, encoding='utf-8').readlines()[:sample_size]

    print("Normalizing...")
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s1), normalize_string(s2)] for s1, s2 in zip(lines1, lines2)]

    input_lang = Lang("en")
    output_lang = Lang("de")

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(path_en, path_de, sample_size=-1):
    input_lang, output_lang, pairs = read_langs(path_en, path_de, sample_size)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else 2 for word in sentence.split(' ')]
    # return lang.tokenizer.encode(sentence, add_special_tokens=False)


def tensor_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang, device):
    input_tensor = tensor_from_sentence(input_lang, pair[0], device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], device)
    return input_tensor, target_tensor
