from data.CharacterLevel import *
from model.rnn import RNNSimple
import torch.nn as nn
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden().to(device)

    rnn.zero_grad()

    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_data():
    category_lines = {}
    all_categories = []

    for file in findfiles('dataset/data/names/*.txt'):
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        lines_file = read_lines(file)
        category_lines[category] = lines_file

    return category_lines, all_categories


# Just return an output given a line
def evaluate(rnn, line_tensor):
    hidden = rnn.init_hidden().to(device)

    output = None

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "./output"

    category_lines, all_categories = get_data()
    n_categories = len(all_categories)

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    rnn = RNNSimple(n_letters, hidden_size=128, output_size=n_categories).to(device)

    criterion = nn.NLLLoss().to(device)

    learning_rate = 0.005

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output, loss = train(category_tensor.to(device), line_tensor.to(device))
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Plotting the results
    plt.figure()
    plt.plot(all_losses)
    plt.savefig(os.path.join(output_dir, "CharacterLevelDemo_loss.png"))

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate(rnn, line_tensor.to(device))
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(os.path.join(output_dir, "CharacterLevelDemo_confusion_matrix.png"))

