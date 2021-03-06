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
                 bpe=True, target=None, word2id=None):
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
        """
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.bpe = bpe
        self.word2id = word2id

        if target != None:
            special_tkns.append(target)
        self.target = target

        if bpe:
            eng_lines = []
            frn_lines = []
            with open(input_file, encoding="utf-8") as f:
                raw = unicodedata.normalize("NFKC", f.read().strip()).split("\n")
                for line in raw:
                    assert len(line.split("\t")) == 2
                    eng_phrase, frn_phrase = line.split("\t")
                    eng_lines.append(eng_phrase.split())
                    frn_lines.append(frn_phrase.split())

        else:
            eng_lines, frn_lines = read_from_corpus(input_file)

        assert len(eng_lines) == len(frn_lines)
        self.length = len(eng_lines)

        if bpe:
            self.word2id = self.get_vocab2id(eng_lines + frn_lines)
            self.pad_id, self.start_id, self.stop_id = self.special_tkn_ids(self.word2id)
            self.vocab_size = len(self.word2id)
            self.output_size = self.vocab_size #the bpe is joint

        else:
            self.eng_word2id = self.get_vocab2id(eng_lines)
            self.frn_word2id = self.get_vocab2id(frn_lines)
            self.pad_id, self.start_id, self.stop_id = self.special_tkn_ids(self.eng_word2id)
            self.vocab_size = len(self.frn_word2id)
            self.output_size = len(self.eng_word2id)

        self.process_lines(frn_lines, eng_lines)
        if self.bpe: #check labes all begin w/ START tag
            assert torch.equal(self.decoder_inputs[:, 0], torch.full([len(self)], self.word2id["START"]))
            assert torch.equal(torch.eq(self.labels[:, 0], torch.full([len(self)], self.word2id["START"])),
                torch.full([len(self)], False))

        else:
            assert torch.equal(self.decoder_inputs[:, 0], torch.full([len(self)], self.frn_word2id["START"]))
            assert torch.equal(torch.eq(self.labels[:, 0], torch.full([len(self)], self.frn_word2id["START"])),
                torch.full([len(self)], False))

    def get_vocab2id(self, parsed_lines):
        vocab = set()
        for line in parsed_lines:
            vocab |= set(line)

        if self.word2id == None:
            vocab2id = {word : id for id,word in enumerate(special_tkns + list(vocab))}
            return vocab2id
        #could speed up mayb by doing set difference with given word2id....
        else:
            new_id = len(self.word2id)
            for word in vocab:
                if word not in self.word2id:
                    self.word2id[word] = new_id
                    new_id += 1
            return self.word2id

    def special_tkn_ids(self, lookup):
        return lookup["PAD"], lookup["START"], lookup["STOP"]

    def process_lines(self, encode_lines, decode_lines):
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.labels = []
        self.encoder_lengths = {}
        self.decoder_lengths = {}

        print("vectorizing data...")
        for i in tqdm(range(len(encode_lines))):
            if self.bpe:
                enc = [self.word2id[e] for e in encode_lines[i]]
                if self.target != None:
                    enc = [self.word2id[self.target]] + enc
                dec = [self.start_id] + [self.word2id[d] for d in decode_lines[i]]

            else:
                enc = [self.frn_word2id[e] for e in encode_lines[i]]
                dec = [self.start_id] + [self.eng_word2id[d] for d in decode_lines[i]]


            if len(enc) >= self.enc_seq_len:
                enc = enc[:self.enc_seq_len-1]
            enc.append(self.stop_id) #id is same in bpe and vanilla
            self.encoder_lengths[i] = len(enc)

            if len(dec) > self.dec_seq_len:
                dec = dec[:self.dec_seq_len]
            self.decoder_lengths[i] = len(dec)
            targ = dec[1:] + [self.stop_id]

            #reversing order should bump accuracy a bit
            #self.encoder_inputs.append(torch.flip(torch.tensor(enc), [0]))
            self.encoder_inputs.append(torch.tensor(enc))
            self.decoder_inputs.append(torch.tensor(dec))
            self.labels.append(torch.tensor(targ))

        self.encoder_inputs = pad_sequence(self.encoder_inputs, True, self.pad_id)
        self.decoder_inputs = pad_sequence(self.decoder_inputs, True, self.pad_id)
        self.labels = pad_sequence(self.labels, True, self.pad_id)

        #when all sequences < rnn size
        if self.encoder_inputs.size()[1] != self.enc_seq_len:
            self.encoder_inputs = torch.cat((self.encoder_inputs, torch.full([self.encoder_inputs.size()[0],
                self.enc_seq_len - self.encoder_inputs.size()[1]], self.pad_id)), 1)

        if self.decoder_inputs.size()[1] != self.dec_seq_len:
            self.decoder_inputs = torch.cat((self.decoder_inputs, torch.full([self.decoder_inputs.size()[0],
                self.dec_seq_len - self.decoder_inputs.size()[1]], self.pad_id)), 1)

            self.labels = torch.cat((self.labels, torch.full([self.labels.size()[0],
                self.dec_seq_len - self.labels.size()[1]], self.pad_id)), 1)


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

def learn_bpe(train_file, iterations):
    """
    learn_bpe learns the BPE from data in the train_file and return a
    dictionary of {Byte Pair Encoded vocabulary: count}.

    Note: The original vocabulary should not include '</w>' symbols.
    Note: Make sure you use unicodedata.normalize to normalize the strings when
          reading file inputs.

    You are allowed to add helpers.

    :param train_file: file of the original version
    :param iterations: number of iterations of BPE to perform

    :return: vocabulary dictionary learned using BPE
    """
    # TODO: Please implement the BPE algorithm.
    vocab = get_frequency_vocab(train_file)
    for i in range(iterations):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    return vocab


def get_frequency_vocab(input_file):
    vocab = {}
    with open(input_file) as f:
        raw = unicodedata.normalize("NFKC", f.read().strip()).split("\n")
        lines = [line for line in raw]

    for line in lines:
        spaced_words = [" ".join(word) for word in line.split()]
        for spaced_word in spaced_words:
            if spaced_word in vocab:
                vocab[spaced_word] += 1

            else:
                vocab[spaced_word] = 1

    return vocab


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out



def get_transforms(vocab):
    """
    get_transforms return a mapping from an unprocessed vocabulary to its Byte
    Pair Encoded counterpart.

    :param vocab: BPE vocabulary dictionary of {bpe-vocab: count}

    :return: dictionary of {original: bpe-version}
    """
    transforms = {}
    for vocab, count in vocab.items():
        word = vocab.replace(' ', '')
        bpe = vocab + " </w>"
        transforms[word] = bpe
    return transforms


def apply_bpe(train_file, bpe_file, vocab):
    """
    apply_bpe applies the BPE vocabulary learned from the train_file to itself
    and save it to bpe_file.

    :param train_file: file of the original version
    :param bpe_file: file to save the Byte Pair Encoded version
    :param vocab: vocabulary dictionary learned from learn_bpe
    """
    with open(train_file) as r, open(bpe_file, 'w', encoding="utf-8") as w:
        transforms = get_transforms(vocab)
        for line in r:
            line = unicodedata.normalize("NFKC", line)
            words = re.split(r'(\s+)', line.strip())
            bpe_str = ""
            for word in words:
                if word.isspace():
                    bpe_str += word
                else:
                    bpe_str += transforms[word]
            bpe_str += "\n"
            w.write(bpe_str)


def count_vocabs(eng_lines, frn_lines):
    eng_vocab = defaultdict(lambda: 0)
    frn_vocab = defaultdict(lambda: 0)

    for eng_line in eng_lines:
        for eng_word in eng_line:
            eng_vocab[eng_word] += 1
    for frn_line in frn_lines:
        for frn_word in frn_line:
            frn_vocab[frn_word] += 1

    return eng_vocab, frn_vocab


def read_from_corpus(corpus_file):
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())
    return eng_lines, frn_lines


def unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold=5):
    for eng_line in eng_lines:
        for i in range(len(eng_line)):
            if eng_vocab[eng_line[i]] <= threshold:
                eng_line[i] = "UNK"

    for frn_line in frn_lines:
        for i in range(len(frn_line)):
            if frn_vocab[frn_line[i]] <= threshold:
                frn_line[i] = "UNK"


def preprocess_vanilla(corpus_file, threshold=5):
    """
    preprocess_vanilla unks the corpus and returns two lists of lists of words.

    :param corpus_file: file of the corpus
    :param threshold: threshold count to UNK
    """
    eng_lines, frn_lines = read_from_corpus(corpus_file)
    eng_vocab, frn_vocab = count_vocabs(eng_lines, frn_lines)
    unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold)
    return eng_lines, frn_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()

    vocab = learn_bpe(args.input_file, args.iterations)
    apply_bpe(args.input_file, args.output_file, vocab)

    """
    dataset = TranslationDataset('data/bpe-eng-fraS.txt', 25, 25)
    #dataset = TranslationDataset('data/joint-bpe-eng-deuS.txt', 25, 25, target='<2en>')
    print(dataset.__len__())
    print()
    if dataset.target != None:
        print(dataset.word2id[dataset.target])
    print(dataset.__getitem__(dataset.__len__()-1)["dec_input"])
    print(dataset.__getitem__(dataset.__len__()-1)["enc_input"])
    """
