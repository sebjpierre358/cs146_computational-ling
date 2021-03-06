from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import argparse
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

special_tkns = ["PAD", "START", "STOP"]

class TranslationDataset(Dataset):
    def __init__(self, input_file, enc_seq_len, dec_seq_len,
                 bpe=True, target=None, word2id=None, flip=False):
        """
        Read and parse the translation dataset line by line. Make sure you
        separate them on tab to get the original sentence and the target
        sentence. You will need to adjust implementation details such as the
        vocabulary depending on whether the dataset is BPE or not.

        :param input_file: the data file pathname
        :param enc_seq_len: sequence length of encoder
        :param dec_seq_len: sequence length of decoder
        :param bpe: whether this is a Byte Pair Encoded dataset
        :param target: the tag for target language
        :param word2id: the word2id to append upon
        :param flip: whether to flip the ordering of the sentences in each line
        """
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.bpe = bpe
        self.word2id = word2id

        if target != None:
            special_tkns.append(target)
        self.target = target

        lang_1_lines, lang_2_lines = read_from_corpus(input_file, flip)

        assert len(lang_1_lines) == len(lang_2_lines)
        self.length = len(lang_1_lines)

        self.word2id = self.get_vocab2id(lang_1_lines + lang_2_lines)
        self.pad_id, self.start_id, self.stop_id = self.special_tkn_ids(self.word2id)
        self.vocab_size = len(self.word2id)
        #assuming the bpe is joint
        self.output_size = self.vocab_size

        (self.encoder_inputs,
        self.decoder_inputs,
        self.labels,
        self.encoder_lengths,
        self.decoder_lengths) = process_lines(lang_1_lines, lang_2_lines, self.word2id, enc_seq_len, self.target)

    def get_vocab2id(self, parsed_lines):
        vocab = set()
        for line in parsed_lines:
            vocab |= set(line)

        if self.word2id == None:
            vocab2id = {word : id for id,word in enumerate(special_tkns + list(vocab))}
            return vocab2id

        else:
            if self.target not in self.word2id:
                self.word2id[self.target] = len(self.word2id)
            new_id = len(self.word2id)
            for word in vocab:
                if word not in self.word2id:
                    self.word2id[word] = new_id
                    new_id += 1
            return self.word2id

    def special_tkn_ids(self, lookup):
        return lookup["PAD"], lookup["START"], lookup["STOP"]

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
        item = {"enc_input" : self.encoder_inputs[idx],
                "dec_input" : self.decoder_inputs[idx],
                "labels" : self.labels[idx],
                "enc_len" : torch.tensor(self.encoder_lengths[idx]),
                "dec_len" : torch.tensor(self.decoder_lengths[idx])}

        return item

def process_lines(encode_lines, decode_lines, word2id, max_seq_len, target=None):
    encoder_inputs = []
    decoder_inputs = []
    labels = []
    encoder_lengths = {}
    decoder_lengths = {}
    start_id = word2id["START"]
    stop_id = word2id["STOP"]
    pad_id = word2id["PAD"]

    print("vectorizing data...")
    #print(encode_lines[:5])
    #print(decode_lines[:5])
    for i in tqdm(range(len(encode_lines))):
        enc = [word2id[e] for e in encode_lines[i]]
        if target != None:
            enc = [word2id[target]] + enc
        dec = [start_id] + [word2id[d] for d in decode_lines[i]]

        if len(enc) >= max_seq_len:
            enc = enc[:max_seq_len-1]
        enc.append(stop_id)
        encoder_lengths[i] = len(enc)

        if len(dec) > max_seq_len:
            dec = dec[:max_seq_len]
        decoder_lengths[i] = len(dec)
        targ = dec[1:] + [stop_id]

        #reversing order should bump accuracy a bit
        #encoder_inputs.append(torch.flip(torch.tensor(enc), [0]))
        encoder_inputs.append(torch.tensor(enc))
        decoder_inputs.append(torch.tensor(dec))
        labels.append(torch.tensor(targ))

    encoder_inputs = pad_sequence(encoder_inputs, True, pad_id)
    decoder_inputs = pad_sequence(decoder_inputs, True, pad_id)
    labels = pad_sequence(labels, True, pad_id)

    #when all sequences < rnn size
    if encoder_inputs.size()[1] != max_seq_len:
        encoder_inputs = torch.cat((encoder_inputs, torch.full([encoder_inputs.size()[0],
            max_seq_len - encoder_inputs.size()[1]], pad_id)), 1)

    if decoder_inputs.size()[1] != max_seq_len:
        decoder_inputs = torch.cat((decoder_inputs, torch.full([decoder_inputs.size()[0],
            max_seq_len - decoder_inputs.size()[1]], pad_id)), 1)

        labels = torch.cat((labels, torch.full([labels.size()[0],
            max_seq_len - labels.size()[1]], pad_id)), 1)

    return encoder_inputs, decoder_inputs, labels, encoder_lengths, decoder_lengths


def read_from_corpus(corpus_file, flip, for_zshot=False):
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = unicodedata.normalize("NFKC", line)
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            #eng_line = re.sub('_ ', ' _ ', eng_line)
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)
            #frn_line = re.sub('_ ', ' _ ', frn_line)

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())

    if for_zshot:
        if flip:
            return frn_lines
        else:
            return eng_lines

    else:
        if flip:
            return frn_lines, eng_lines
        else:
            return eng_lines, frn_lines


class ZeroshotDataset(Dataset):
    def __init__(self, enc_file, dec_file, target, word2id, max_seq_len, flip=True):
        """
        Testing dataset composed by pairing the inputs from the preprocessed
        X -> Y translation test corpus with the Y -> Z corpus for testing zeroshot
        translation from language X -> Z

        :param enc_file: path to test file containing the source language
        :param dec_file: path to the test file containing the target language
        :param target: tag id for the target language
        :param word2id: word/id dict over bpe vocab associated with the two train files
        :param max_seq_len: maximum length of a sequence. should match that of train dataset
        :param flip: pull the second language in the enc_file if True
        """
        self.word2id = word2id
        self.target = target

        lang_1_lines = read_from_corpus(enc_file, flip, True)
        #assume that dec_file has the languages as Y -> Z
        lang_2_lines = read_from_corpus(dec_file, True, True)

        assert len(lang_1_lines) == len(lang_2_lines)
        self.length = len(lang_1_lines)

        self.update_vocab2id(lang_1_lines + lang_2_lines)
        self.pad_id, self.start_id, self.stop_id = self.special_tkn_ids(self.word2id)
        self.vocab_size = len(self.word2id)
        self.output_size = self.vocab_size


        (self.enc_inputs,
        self.dec_inputs,
        self.labels,
        self.encoder_lengths,
        self.decoder_lengths) = process_lines(lang_1_lines, lang_2_lines, self.word2id, max_seq_len, self.target)

    def update_vocab2id(self, parsed_lines):
        vocab = set()
        for line in parsed_lines:
            vocab |= set(line)

        if self.target not in self.word2id:
            self.word2id[self.target] = len(self.word2id)

        new_id = len(self.word2id)
        for word in vocab:
            if word not in self.word2id:
                self.word2id[word] = new_id
                new_id += 1



    def special_tkn_ids(self, lookup):
        return lookup["PAD"], lookup["START"], lookup["STOP"]

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return self.labels.size()[0]

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        item = {"enc_input" : self.enc_inputs[idx],
                "dec_input" : self.dec_inputs[idx],
                "labels" : self.labels[idx],
                "enc_len" : torch.tensor(self.encoder_lengths[idx]),
                "dec_len" : torch.tensor(self.decoder_lengths[idx])}

        return item



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()
    """

    dset = TranslationDataset("data/oneShotDEFR/dattrainED.txt", 30, 30, target='<2deu>', flip=True)
    dset2 = TranslationDataset("data/oneShotDEFR/dattrainEF.txt", 30, 30, target='<2fr>', word2id=dset.word2id)
    #print(dset[6])
    zero = ZeroshotDataset("data/oneShotDEFR/dattestED.txt", "data/oneShotDEFR/dattestEF.txt",
        '<2fr>', dset2.word2id, 30)
    #    targs, dset.encoder_lengths, dset.decoder_lengths)
    #ok = dset[5]
    #print(ok)
    """
