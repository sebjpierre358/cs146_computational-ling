from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence

def load_dataset(fn_tr, fn_tst, tokenizer, batch_size, window_size):
    """
    :param fn_tr: filename for the training dataset
    :param fn_tst: filename for the testing dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
    So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
    the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
    """
    #if tokenizer == None:
    #    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #tokenizer.pad_token = tokenizer.eos_token

    dset = LanguageModelDataset(fn_tr, window_size, tokenizer)
    test_set = LanguageModelDataset(fn_tst, window_size, tokenizer, False)

    train_loader = DataLoader(dset, batch_size=batch_size,
                              shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, pin_memory=True)

    return train_loader, test_loader

class LanguageModelDataset(Dataset):
    def __init__(self, input_file, window_size, gpt_tokenizer, shuffle=True):
        """
        Read and parse the dataset for the transformer line by line.

        :param input_file: the data file pathname
        :param window_size: max sequence length of inputs
        :gpt_tokenizer: pretrained gpt tokenizer to turn words into id vectors
        :param shuffle: whether the lines should be shuffled
        """
        self.window_size = window_size
        self.tokenizer = gpt_tokenizer
        self.inputs, self.masks, self.inputs_cut, self.labels = corpus2vec(input_file, window_size, self.tokenizer, shuffle)

        self.length = self.inputs.shape[0]

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
        item = {"gpt_inputs" : self.inputs[idx],
                "inputs" : self.inputs_cut[idx],
                "mask" : self.masks[idx],
                "labels" : self.labels[idx]}

        return item


def corpus2vec(corpus_file, window_size, tokenizer, shuffle=True):
    lines = []
    with open(corpus_file, 'r', encoding="utf-8") as f:
        print("tokenizing corpus.....")
        for line in tqdm(f):
            if line == "STOP":
                continue
            else:
                line = tokenizer(line, max_length=window_size+1, truncation=True)

                lines.append(line)

    if shuffle:
        random.shuffle(lines)

    inputs = pad_sequence(list(map(lambda x: torch.LongTensor(x['input_ids']), lines)), True, 0.0)
    attn_masks = pad_sequence(list(map(lambda x: torch.FloatTensor(x['attention_mask']), lines)), True, 0.0)
    inputs_cut = pad_sequence(list(map(lambda x: torch.LongTensor(x['input_ids'][:-1]), lines)), True)
    labels = pad_sequence(list(map(lambda x: torch.LongTensor(x['input_ids'][1:]), lines)), True)

    return inputs, attn_masks, inputs_cut, labels
