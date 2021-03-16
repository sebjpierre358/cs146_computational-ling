from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import re
from tqdm import tqdm 
import random

def load_dataset(fn_tr, fn_tst, window_size, batch_sz):
    """
    :param fn_tr: filename for the training dataset
    :param fn_tst: filename for the testing dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, vocab_size) for train and test
    :Comment: You don't have to shuffle the test dataset
    """
    train_set = ParsingDataset(fn_tr, window_size)
    test_set = ParsingDataset(fn_tst, window_size, train_set.word2id, False)
    train_set.update_word2id(test_set.word2id)

    train_loader = DataLoader(train_set, batch_size=batch_sz,
                              shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=batch_sz,
                              shuffle=False, pin_memory=True)

    return (train_loader, test_loader, test_set.vocab_size)

class ParsingDataset(Dataset):
    def __init__(self, input_file, window_size, word2id=None, shuffle=True):
        """
        Read and parse the dataset for the transformer line by line.

        :param input_file: the data file pathname
        :param word2id: the word2id to append upon
        :param shuffle: whether the lines should be shuffled
        """
        self.window_size = window_size
        self.word2id = word2id
        lines = read_from_corpus(input_file)
        if shuffle:
            random.shuffle(lines)

        self.length = len(lines)

        self.word2id = self.get_vocab2id(lines)
        self.pad_id = self.word2id['PAD']
        self.vocab_size = len(self.word2id)

        self.inputs, self.labels = process_lines(lines, self.word2id, self.window_size)

    def get_vocab2id(self, parsed_lines):
        vocab = set()
        for line in parsed_lines:
            vocab |= set(line)

        if self.word2id == None:
            vocab2id = {word : id for id,word in enumerate(['PAD'] + list(vocab))}
            return vocab2id

        else:
            new_id = len(self.word2id)
            for word in vocab:
                if word not in self.word2id:
                    self.word2id[word] = new_id
                    new_id += 1
            return self.word2id

    def update_word2id(self, new_word2id):
        self.word2id = new_word2id
        self.vocab_size = len(new_word2id)

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.length

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {"input" : self.inputs[idx],
                "label" : self.labels[idx]}

        return item


def read_from_corpus(corpus_file):
    lines = []
    with open(corpus_file, 'r', encoding="utf-8") as f:
        for line in f:
            if line == "STOP":
                continue
            else:
                line = re.sub('([.,!?():;])', r' \1 ', line)
                line = re.sub('\s{2,}', ' ', line)

                lines.append(line.split())

    return lines

def process_lines(lines, word2id, window_size):
    inputs = []
    labels = []
    stop_id = word2id["STOP"]
    pad_id = word2id["PAD"]

    print("vectorizing data...")
    for i in tqdm(range(len(lines))):
        seq = [word2id[w] for w in lines[i]]

        if len(seq) > window_size+1:
            seq = seq[:window_size] + [stop_id]
        input = seq[:-1]
        label = seq[1:]

        inputs.append(torch.LongTensor(input))
        labels.append(torch.LongTensor(label))

    inputs = pad_sequence(inputs, True, pad_id)
    labels = pad_sequence(labels, True, pad_id)

    #when all sequences < window size
    if inputs.size()[1] != window_size:
        inputs = torch.cat((inputs, torch.full([inputs.size()[0],
            window_size - inputs.size()[1]], pad_id)), 1)

        labels = torch.cat((labels, torch.full([labels.size()[0],
            window_size - labels.size()[1]], pad_id)), 1)

    return inputs, labels
