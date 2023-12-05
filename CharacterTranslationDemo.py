import functools
import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from data.CharacterTranslation import tensor_from_sentence, prepare_data, get_dataloader
from model.rnn import EncoderRNN, DecoderRNN
from model.attentionrnn import AttnDecoderRNN

import os

SOS_token = 0
EOS_token = 1


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def showPlot(points, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    output_dir = "./output"
    plt.savefig(f"{output_dir}/CharacterTranslationDemo_loss_{name}.png")


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100, name="A"):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % ((time_since(start, epoch / n_epochs)),
                  epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses, name)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_randomly(encoder, decoder, n=10):
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def show_attention(input_sentence, output_words, attentions, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    output_dir = "./output"
    file_name = f"CharacterTranslationDemo_attention_{name}.png"

    plt.savefig(os.path.join(output_dir, file_name))


def evaluate_and_show_attention(input_sentence, encoder, decoder, input_lang, output_lang, name):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions[0, :len(output_words), :], name=name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    decoder_att = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    print("Start training Decoder...")
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5, name="no_attention")

    print("Start training Attention...")
    train(train_dataloader, encoder, decoder_att, 80, print_every=5, plot_every=5, name="Attention")

    encoder.eval()
    decoder_att.eval()
    evaluate_randomly(encoder, decoder_att)

    evaluate_and_show_attention_fn = functools.partial(
        evaluate_and_show_attention,
        encoder=encoder,
        decoder=decoder_att,
        input_lang=input_lang,
        output_lang=output_lang
    )

    evaluate_and_show_attention_fn('il n est pas aussi grand que son pere', name="A")

    evaluate_and_show_attention_fn('je suis trop fatigue pour conduire', name="B")

    evaluate_and_show_attention_fn('je suis desole si c est une question idiote', name="C")

    evaluate_and_show_attention_fn('je suis reellement fiere de vous', name="D")


if __name__ == '__main__':
    main()
