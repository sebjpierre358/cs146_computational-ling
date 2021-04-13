from torch.utils.data import Dataset
import torch
import numpy as np
import random
from collections import OrderedDict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self, file_path, win_sz, msk_ratio=.15, rand_seed=None, word2id=None):
        super().__init__()
        self.rng = np.random.default_rng(rand_seed)
        self.msk_ratio=msk_ratio
        self.win_sz = win_sz

        content = read_file(file_path)
        #print(content.index('bank'))
        #print("num occurences: {}".format(content.count('bank')))
        self.build_word2id(content, word2id)

        num_inputs = int(len(content) / win_sz) + 1
        self.raw_inputs = np.array_split(np.array(content), num_inputs)
        #print(self.raw_inputs.shape)
        random.shuffle(self.raw_inputs)
        self.length = len(self.raw_inputs)
        self.vocab_size = len(self.word2id)

        self.process_lines(self.raw_inputs)

    def build_word2id(self, parsed_text, word2id):
        #using an ordered dict maintains the id of each token in the corpus
        # each time the corpus file is parsed. Avoids having to give BERT a word2id
        # dict when it's loaded
        vocab = OrderedDict.fromkeys(parsed_text)

        if word2id == None:
            self.word2id = {word : id for id,word in enumerate(['PAD'] + ['MASK'] + list(vocab.keys()))}

        else:
            new_id = len(word2id)
            for word in vocab:
                if word not in word2id:
                    self.word2id[word] = new_id
                    new_id += 1

            self.word2id = word2id

    def update_word2id(self, new_word2id):
        self.word2id = new_word2id
        self.vocab_size = len(new_word2id)

    def mask_sequence(self, seq):
        seq_len = len(seq)
        masked_sz = int(seq_len * self.msk_ratio)
        msk_indices = set(self.rng.choice(seq_len, masked_sz, replace=False))
        rands = self.rng.random(masked_sz).tolist()
        for idx in msk_indices:
            z = rands.pop()
            if z < .8:
                seq[idx] = self.word2id["MASK"]
            else:
                if z > .9:
                    #keep this word for prediction unchanged
                    continue
                # don't want to randomly replace with either PAD or MASK)
                seq[idx] = self.rng.choice(np.arange(2, len(self.word2id)))

        return msk_indices

    def process_lines(self, raw_inputs):
        input_tensors = []
        labels = []
        win_indices = np.arange(self.win_sz)
        pad_id = self.word2id['PAD']

        print("vectorizing & masking data.....")
        for i in tqdm(range(len(raw_inputs))):
            sequence = raw_inputs[i]
            r = self.rng.random()
            #if r > .9:
            #    print(sequence)
            #    print()
            input = np.vectorize(self.word2id.get)(sequence)
            if len(sequence) != self.win_sz:
                seq_len = self.win_sz
            else:
                seq_len = len(sequence)

            msk_indices = self.mask_sequence(input)
            label = np.zeros(len(sequence))
            for mask_idx in msk_indices:
                label[mask_idx] = self.word2id[sequence[mask_idx]]

            input_tensors.append(torch.LongTensor(input))
            labels.append(torch.LongTensor(label))
            #if r > .9:
            #    print(input_tensors[-1])
            #    print()
            #    print(labels[-1])
            #    print(sequence)
            #    print("---------------------------------")
            #    print()

        self.inputs = pad_sequence(input_tensors, True, pad_id)
        self.labels = pad_sequence(labels, True, pad_id)


    def get_instances(self, word_list):
        """
        Given a word and its possible derivatives (eg: work, works, worked), returns
        all examples in the data where the given words appear

        Inputs:
        - word_list: list of words to gather the examples for
        Return:
        - a tuple of
         LongTensor of all examples in the data where the words in word_list
          appear
         - a np array of all indices where the word appears in each
          example
         - a list of the relevant words with respect to each example
        """
        word_ids = {self.word2id[w] : w for w in word_list}
        #$print(word_ids)
        examples = []
        indices = []
        word_order = []
        def word_in_seq(x):
            x_asid = np.vectorize(self.word2id.get)(x)
            for id in word_ids:
                num_present = np.sum(np.isin(x_asid, [id]))
                if num_present != 0:
                    word_indices = np.where(x_asid == id)[0]
                    #print(x_asid)
                    #print(np.where(x_asid == id))
                    for i in range(len(word_indices)):
                        examples.append(torch.LongTensor(x_asid))
                        indices.append(word_indices[i])
                        word_order.append(word_ids[id])

        for seq in self.raw_inputs:
            word_in_seq(seq)

        examples = pad_sequence(examples, True, self.word2id["PAD"])
        #print(examples[0])
        indices = np.array(indices)

        return examples, indices, word_order


    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.length

    def __getitem__(self, i):
        """
        __getitem__ should return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {"input" : self.inputs[i],
                "label" : self.labels[i]}

        return item


def read_file(fname):
    content = []
    with open(fname) as f:
        for line in f.readlines():
            content += line.lower().strip().split()[:-1] + ['STOP']
            #need to consider the STOP token different from 'stop' word
            #assumes every line ends with STOP
    return content

"""
dset2 = MyDataset('data/penn-UNK-test.txt', 50)
#dset = MyDataset('data/penn-UNK-train.txt', 20)
#print(len(dset))
#dset[len(dset)-1]
#it = dset[235]
#print(dset.word2id['MASK'])
#print(it['input'])
#print(it['label'])
print(dset2.word2id['bank'])
print(dset2.word2id['banks'])
#tried, tried_idx, w_order = dset2.get_instances(['bank', 'banks'])
#print(tried)
#print(tried_idx)
#print((w_order))
"""
