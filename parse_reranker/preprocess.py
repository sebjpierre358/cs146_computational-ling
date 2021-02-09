from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import random


class ParsingDataset(Dataset):
    def __init__(self, input_file):
        """
        Read and parse the train file line by line. Create a vocabulary
        dictionary that maps all the unique tokens from your data as
        keys to a unique integer value. Then vectorize your
        data based on your vocabulary dictionary.
        :param input_file: the data file pathname
        """
        with open(input_file) as f:
            raw = f.read().strip().split("\n")
            lines = [line for line in raw]

        random.shuffle(lines)

        longest_line = 0
        vocab = set()
        parses = []
        self.parse_lengths = []
        for line in lines:
            parses.append(line.split())
            if len(parses[-1]) == 1:
                parses.pop()
                continue
            #inputs and labels will always be 1 token less than the full original parse
            self.parse_lengths.append(len(parses[-1]) - 1)
            vocab |= {token for token in parses[-1]}

        split_index = int(len(parses) * .9) #for splitting into train/validate sets
        self.avg_len = int(np.mean(self.parse_lengths)) #this will be the model's window size

        for p in range(len(parses)):
            if len(parses[p]) > self.avg_len:
                parses[p] = parses[p][:self.avg_len] + ["STOP"]
                self.parse_lengths[p] = self.avg_len

        vocab.add("PAD")
        vocab = list(vocab)
        self.vocab_size = len(vocab)

        self.word2id = {}
        self.id2word = {}

        for x in range(len(vocab)):
            self.word2id[vocab[x]] = x
            self.id2word[x] = vocab[x]

        self.pad_idx = self.word2id["PAD"]

        self.train_indices = [i for i in range(split_index)]
        assert len(parses) == len(self.parse_lengths)

        self.validate_indices = [j for j in range(split_index, len(parses))]

        def vectorize(data):
            vectorized = []
            for d in data:
                vectorized.append([self.word2id[c] for c in d])
            return vectorized

        def pad(data):
            inputs = []
            labels = []
            for sequence in data:
                inputs.append(sequence[:-1] + ["PAD"] * (self.avg_len-len(sequence)+1))
                labels.append(sequence[1:] + ["PAD"] * (self.avg_len-len(sequence)+1))
            return inputs, labels

        padded_data = pad(parses)
        self.input_ids = vectorize(padded_data[0])
        self.label_ids = vectorize(padded_data[1])

        assert len(self.input_ids[0]) == self.avg_len

        self.dataset_len = len(parses)

    def wordify(self, data):
        worded = []
        for d in data:
            worded.append(self.id2word[d])
        return worded


    def __len__(self):
        """
        len should return a the length of the dataset
        :return: an integer length of the dataset
        """
        return self.dataset_len

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        :param idx: the index for retrieval
        :return: tuple or dictionary of the data
        """
        item = {"parse": torch.tensor(self.input_ids[idx]),
                "label": torch.tensor(self.label_ids[idx]),
                "length": torch.tensor(self.parse_lengths[idx])}

        """
        item = {"parse": torch.tensor(self.train_input_ids[idx]),
                "label": torch.tensor(self.train_label_ids[idx]),
                "length": torch.tensor(self.train_lengths[idx])}
        """

        return item


class RerankingDataset(Dataset):
    def __init__(self, parse_file, gold_file, word2id, window_size):
        """
        Read and parse the parse files line by line. Unk all words that has not
        been encountered before (not in word2id). Split the data according to
        gold file. Calculate number of constituents from the gold file.
        :param parse_file: the file containing potential parses
        :param gold_file: the file containing the right parsings
        :param word2id: the previous mapping (dictionary) from word to its word
                        id
        """
        with open(parse_file) as f:
            raw = f.read().strip().split("\n")
            lines = [line for line in raw]

        """
        def tensor_builder(trees):

            :param trees: a list of lists containing the parse trees for a particular
            sentence
            :return: the list of lists as an int tensor

            new_tensor = None
            for tree in trees:
                if new_tensor != None:
                    print(new_tensor)
                    print()
                    new_tensor = torch.cat((new_tensor, torch.tensor(tree)), 0)
                    print(new_tensor)
                    print("------------------------")
                    print()

                else:
                    new_tensor = torch.tensor(tree)

            #print(new_tensor.size())
            return new_tensor
        """


        self.sentences = {}
        self.num_sentence = -1
        for line in lines:
            split_line = line.split()
            if len(split_line) == 1: #start of new parse tree set
                self.num_sentence += 1
                self.sentences[self.num_sentence] = []
                if self.num_sentence != 0:
                    test_tensor = torch.tensor(self.sentences[self.num_sentence-1])
                    self.sentences[self.num_sentence-1] = test_tensor

            else:
                row = [] #3rd item corresponds to last token of tree (for testing)
                row.append(int(split_line[0])) #num correct constits
                row.append(int(split_line[1])) #total num constits
                tree = split_line[2:]

                #add input length & last token as 3rd & 4th items in row respectively
                if len(tree) > window_size + 1:
                    tree = tree[:window_size+1]
                    row.append(window_size)
                    row.append(word2id["STOP"])

                else:
                    row.append(len(tree) - 1) #last token does not constitute input to model
                    if tree[-1] not in word2id:
                        row.append(word2id["*UNK"])

                    else:
                        row.append(word2id[tree[-1]])

                vectored = [] #will not include last token of full parse
                for token in tree[:-1]:
                    if token not in word2id:
                        vectored.append(word2id["*UNK"])
                    else:
                        vectored.append(word2id[token])

                #we already accounted for case where tree > window_size
                if len(vectored) < window_size:
                    vectored = vectored + [word2id["PAD"]] * (window_size-len(vectored))

                row = row + vectored
                self.sentences[self.num_sentence].append(row)

        #due to quirky way of iterating to convert nested list of parse trees to tensor above,
        #very last sentence does not have its trees converted to tensor
        last_tensor = torch.tensor(self.sentences[self.num_sentence])
        self.sentences[self.num_sentence] = last_tensor

        with open(gold_file) as g:
            raw = g.read().strip().split("\n")
            lines = [line for line in raw]

        self.golds = []
        for line in lines:
            self.golds.append((line.count("("), line.split()))

        assert len(self.golds) == (self.num_sentence + 1)

    def __len__(self):
        """
        len should return a the length of the dataset
        :return: an integer length of the dataset
        """
        return self.num_sentence + 1

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        :param idx: the index for retrieval
        :return: tuple or dictionary of the data
        """
        item = {"trees": self.sentences[idx], #dict to tensor of all parse trees of a sentence (+ extra info)
                "gold": int(self.golds[idx][0])} #just the # of constits in gold parse

        return item
