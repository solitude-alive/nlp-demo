import builtins
import torch
import torchtext
from torch.utils.data import DataLoader
import collections
import os
from data.adnews import load_dataset
from model.rnn import RNNClassifier
from model.lstm import LSTMClassifier, LSTMPackClassifier
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = None
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

stoi_hash = {}


def encode(x, voc=None, unk=0, tokenizer=tokenizer):
    global stoi_hash
    v = vocab if voc is None else voc
    if v in stoi_hash.keys():
        stoi = stoi_hash[v]
    else:
        stoi = v.get_stoi()
        stoi_hash[v] = stoi
    return [stoi.get(s, unk) for s in tokenizer(x)]


def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None,
                report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, features in dataloader:
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        out = net(features)
        loss = loss_fn(out, labels)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}")
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count


def padify(b, voc=None, tokenizer=tokenizer):
    # b is the list of tuples of length batch_size
    #   - first element of a tuple = label,
    #   - second = feature (text sequence)
    # build vectorized sequence
    v = [encode(x[1], voc=voc, tokenizer=tokenizer) for x in b]
    # compute max length of a sequence in this minibatch
    l = max(map(len, v))
    return (  # tuple of two tensors - labels and features
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t), (0, l - len(t)), mode='constant', value=0) for t in v])
    )


def pad_length(b):
    # build vectorized sequence
    v = [encode(x[1]) for x in b]
    # compute max length of a sequence in this minibatch and length sequence itself
    len_seq = list(map(len, v))
    l = max(len_seq)
    return (  # tuple of three tensors - labels, padded features, length sequence
        torch.LongTensor([t[0] - 1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t), (0, l - len(t)), mode='constant', value=0) for t in v]),
        torch.tensor(len_seq)
    )


def offsetify(b, voc=None):
    # first, compute data tensor from all sequences
    x = [torch.tensor(encode(t[1], voc=voc)) for t in b]
    # now, compute the offsets by accumulating the tensor of sequence lengths
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0] - 1 for t in b]),  # labels
        torch.cat(x),  # text
        o
    )


def train_epoch_emb(net, dataloader, lr=0.01, optimizer=None, loss_fn=torch.nn.CrossEntropyLoss(), epoch_size=None,
                    report_freq=200, use_pack_sequence=False):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss, acc, count, i = 0, 0, 0, 0
    for labels, text, off in dataloader:
        optimizer.zero_grad()
        labels, text = labels.to(device), text.to(device)
        if use_pack_sequence:
            off = off.to('cpu')
        else:
            off = off.to(device)
        out = net(text, off)
        loss = loss_fn(out, labels)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
        i += 1
        if i % report_freq == 0:
            print(f"{count}: acc={acc.item() / count}")
        if epoch_size and count > epoch_size:
            break
    return total_loss.item() / count, acc.item() / count


if __name__ == '__main__':
    train_dataset, test_dataset, classes, vocab = load_dataset()
    vocab_size = len(vocab)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)
    net_rnn = RNNClassifier(vocab_size, 64, 32, len(classes)).to(device)
    print("Start training RNN model...")
    start_time = time.time()
    train_epoch(net_rnn, train_loader, lr=0.001)
    print("RNN Elapsed time: ", time.time() - start_time, " seconds")

    net_lstm = LSTMClassifier(vocab_size, 64, 32, len(classes)).to(device)
    print("Start training LSTM model...")
    start_time = time.time()
    train_epoch(net_lstm, train_loader, lr=0.001)
    print("LSTM Elapsed time: ", time.time() - start_time, " seconds")

    train_loader_len = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=pad_length, shuffle=True)
    net = LSTMPackClassifier(vocab_size, 64, 32, len(classes)).to(device)
    print("Start training LSTM model with pack sequence...")
    start_time = time.time()
    train_epoch_emb(net, train_loader_len, lr=0.001, use_pack_sequence=True)
    print("LSTM with pack sequence Elapsed time: ", time.time() - start_time, " seconds")
