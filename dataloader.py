from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?\-#]+", r" ", s)
    return s

def readLangs(path_en, path_de, sample_size):
    print("Reading lines...")

    # Read the file and split into lines
    file = open(path_en, encoding='utf-8')
    lines1 = list()
    if sample_size == -1:
        lines1 = file.readlines()
    else:
        for i in range(sample_size):
            lines1.append(file.readline())
    file.close()

    file = open(path_de, encoding='utf-8')
    lines2 = list()
    if sample_size == -1:
        lines2 = file.readlines()
    else:
        for i in range(sample_size):
            lines2.append(file.readline())
    file.close()

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s1), normalizeString(s2)] for s1, s2 in zip(lines1, lines2)]

    input_lang = Lang("en")
    output_lang = Lang("de")

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(path_en, path_de, sample_size=-1):
    en_lang, de_lang, pairs = readLangs(path_en, path_de, sample_size)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        en_lang.addSentence(pair[0])
        de_lang.addSentence(pair[1])
    print("Counted words:")
    print(en_lang.name, en_lang.n_words)
    print(de_lang.name, de_lang.n_words)
    return en_lang, de_lang, pairs


input_lang, output_lang, pairs = prepareData('data/train.en', 'data/train.de', 100)

print(random.choice(pairs))