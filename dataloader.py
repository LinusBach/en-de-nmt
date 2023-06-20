"""
This script is used for data loading and preprocessing. It defines the Lang class, which is used to represent a
language. The Lang class handles tokenization of text data using the HuggingFace Transformers library.
Also, this script includes functions for reading data from text files, tokenizing the data, and filtering sentence pairs
based on a maximum token length.
"""


from io import open
from transformers import AutoTokenizer
import torch
from typing import List, Tuple

de_CLS_token = 3
de_SEP_token = 4


class Lang:
    def __init__(self, name: str, max_length: int = 15):
        self.name = name
        self.max_length = max_length
        if name == "en":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.n_words = 28996
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
            self.n_words = 30000

    def tokenize(self, sentence: str) -> List[int]:
        return self.tokenizer(sentence, add_special_tokens=True, max_length=self.max_length, truncation=True).input_ids

    def tokenize_without_truncation(self, sentence: str) -> List[int]:
        return self.tokenizer(sentence, add_special_tokens=True).input_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def filter_pairs(pairs: List[List[List[int]]], max_length: int) -> Tuple[List[List[int]], List[List[int]]]:
    filtered_pairs = [pair for pair in pairs if len(pair[0]) < max_length and len(pair[1]) < max_length]
    return [pair[0] for pair in filtered_pairs], [pair[1] for pair in filtered_pairs]


def prepare_data(path_en: str, path_de: str, max_length: int, sample_size: int = -1, start_from_sample: int = 0,
                 device: torch.device = "cpu") -> Tuple[Lang, Lang, List[torch.Tensor], List[torch.Tensor]]:
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
