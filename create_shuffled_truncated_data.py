from random import shuffle, seed
from io import open
from dataloader import Lang
import os

max_length = 30
f_en = os.path.join("data", "train_st_" + str(max_length) + ".en")
f_de = os.path.join("data", "train_st_" + str(max_length) + ".de")

input_lang = Lang("en", max_length)
output_lang = Lang("de", max_length)

seed(42)

print("Reading...")
sentences_en = open(os.path.join("data", "train.en"), encoding='utf-8').readlines()
sentences_de = open(os.path.join("data", "train.de"), encoding='utf-8').readlines()
print("Read")

print("Filtering...")
pairs = [[line_english, line_german]
         for line_english, line_german in zip(sentences_en, sentences_de)
         if len(input_lang.tokenize_without_truncation(line_english)) < max_length and
         len(output_lang.tokenize_without_truncation(line_german)) < max_length]
print("Filtered to %s sentence pairs" % len(pairs))
print("Shuffling...")
shuffle(pairs)
print("Shuffled")

with open(f_en, 'w+', encoding='utf-8') as file1:
    with open(f_de, 'w+', encoding='utf-8') as file2:
        for sentences_en, sentence_de in pairs:
            file1.write(sentences_en)
            file2.write(sentence_de)
