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
        # TODO: read the input file line by line and put the lines in a list.
        with open(input_file) as f:
            raw = f.read().strip().split("\n")
            lines = [line for line in raw]

        random.shuffle(lines)

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.
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

        #train_parses = parses[:split_index]
        self.train_indices = [i for i in range(split_index)]
        #train_parses = parses
        #self.train_lengths = self.parse_lengths[:split_index]
        #self.train_lengths = self.parse_lengths
        assert len(parses) == len(self.parse_lengths)
        #assert len(train_parses) == len(self.train_lengths)

        #validate_parses = parses[split_index:]
        self.validate_indices = [j for j in range(split_index, len(parses))]
        #self.validate_lengths = self.parse_lengths[split_index:]

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

        #print(pad(train_parses))
        #padded_training = pad(train_parses)
        padded_data = pad(parses)
        self.input_ids = vectorize(padded_data[0])
        self.label_ids = vectorize(padded_data[1])
        #self.train_input_ids = vectorize(padded_training[0])
        #self.train_label_ids = vectorize(padded_training[1])
        assert len(self.input_ids[0]) == self.avg_len
        #assert len(self.train_input_ids[0]) == self.avg_len
        #self.train_label_ids = vectorize(pad(train_labels))

        #padded_validate = pad(validate_parses)
        #self.validate_input_ids = vectorize(padded_validate[0])
        #self.validate_label_ids = vectorize(padded_validate[1])
        #self.validate_label_ids = vectorize(pad(validate_labels))

        self.dataset_len = len(parses)


        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels
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
        # TODO: Override method to return the items in dataset
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
                    #test_tensor = tensor_builder(self.sentences[self.num_sentence-1])
                    test_tensor = torch.tensor(self.sentences[self.num_sentence-1])
                    self.sentences[self.num_sentence-1] = test_tensor
                    #self.sentences[self.num_sentence-1] = test_tensor

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

                #self.sentences[self.num_sentence].append((num_correct, num_tags, torch.tensor(vectored)))

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
        # TODO: Override method to return length of dataset
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


#parse_data = ParsingDataset("data/reranker_train.txt")
"""
for x in range(parse_data.__len__()):
    item = parse_data.__getitem__(x)
    print("parse/label " + str(x))
    print(parse_data.wordify(item["parse"]))
    print()
    print(parse_data.wordify(item["label"]))
    print()
"""
#test_data = RerankingDataset("data/conv.txt", "data/gold.txt", parse_data.word2id, 61)
#ok = test_data.__getitem__(test_data.__len__()-2)
#print(ok["trees"])

#print(parse_data.parse_lengths.index(0))

#print(parse_data.train_indices)
#print(parse_data.validate_indices)

#print(parse_data.avg_len)
#print()
#item = parse_data.__getitem__(5)
#print(item["parse"])
#print()
#print(item["label"])
#print()
#print(item["length"])
