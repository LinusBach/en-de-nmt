from io import open
from transformers import AutoTokenizer
import torch


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
        return self.tokenizer(sentence, add_special_tokens=True, max_length=self.max_length, truncation=True).input_ids

    def tokenize_without_truncation(self, sentence):
        return self.tokenizer(sentence, add_special_tokens=True).input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def filter_pairs(pairs, max_length):
    filtered_pairs = [pair for pair in pairs if len(pair[0]) < max_length and len(pair[1]) < max_length]
    return [pair[0] for pair in filtered_pairs], [pair[1] for pair in filtered_pairs]


def prepare_data(path_en, path_de, max_length, sample_size=-1, start_from_sample=0, device="cpu"):
    input_lang = Lang("en", max_length)
    output_lang = Lang("de", max_length)

    lines_english = open(path_en, encoding='utf-8').readlines()[start_from_sample:sample_size + start_from_sample]
    lines_german = open(path_de, encoding='utf-8').readlines()[start_from_sample:sample_size + start_from_sample]
    pairs = [[input_lang.tokenize_without_truncation(line_english),
              output_lang.tokenize_without_truncation(line_german)]
             for line_english, line_german in zip(lines_english, lines_german)]

    print("Read %s sentence pairs" % len(pairs))
    en_sequences, de_sequences = filter_pairs(pairs, max_length)
    en_sequences = [torch.LongTensor(sequence).view(-1, 1).to(device) for sequence in en_sequences]
    de_sequences = [torch.LongTensor(sequence).view(-1, 1).to(device) for sequence in de_sequences]
    print("Trimmed to %s sentence pairs based on length of tokenization" % len(en_sequences))

    return input_lang, output_lang, en_sequences, de_sequences
