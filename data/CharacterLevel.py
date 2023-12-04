# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
from io import open
import glob
import os
import unicodedata
import string
import torch


def findfiles(path):
    return glob.glob(path)  # Return a list of paths matching a pathname pattern


all_letters = string.ascii_letters + " .,;'"  # string.ascii_letters: ascii_lowercase + ascii_uppercase
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s: string) -> string:  # fileter out all non-ASCII characters and remove diacritics
    return ''.join(  # ''.join means join the iterable to a string
        c for c in unicodedata.normalize('NFD', s)  # NFD: Normalization Form Canonical Decomposition
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


for file in findfiles('../dataset/character/names/*.txt'):
    # os.path.basename: return the base name of pathname, os.path.splitext: split the path name into a pair root and ext
    category = os.path.splitext(os.path.basename(file))[0]
    all_categories.append(category)
    lines_file = read_lines(file)
    category_lines[category] = lines_file

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


if __name__ == '__main__':
    print(findfiles('../dataset/character/names/*.txt'))

    print(unicode_to_ascii('Ślusàrski'))

    print(letter_to_index('J'))

    print(line_to_tensor('Jones').size())
