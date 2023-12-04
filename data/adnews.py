import builtins
import torch
import torchtext
import collections
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = None
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


def load_dataset(ngrams=1, min_freq=1):
    global vocab, tokenizer
    print("Loading dataset...")
    train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    print('Building vocab...')
    counter = collections.Counter()
    for (label, line) in train_dataset:
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line), ngrams=ngrams))
    vocab = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return train_dataset, test_dataset, classes, vocab
