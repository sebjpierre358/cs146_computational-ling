from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, pad_idx=None):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.
        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.pad_idx = pad_idx

        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size, self.pad_idx)
        #self.gru = torch.nn.GRU(self.embedding_size, self.rnn_size, batch_first=True, num_layers=1)
        self.lstm = torch.nn.LSTM(self.embedding_size, self.rnn_size, batch_first=True, num_layers=1)
        self.fc1 = torch.nn.Linear(self.rnn_size, 500)
        self.fc2 = torch.nn.Linear(500, int(self.vocab_size/2))
        self.fc3 = torch.nn.Linear(int(self.vocab_size/2),self.vocab_size)
        # TODO: initialize embeddings, LSTM, and linear layers

    def forward(self, inputs, lengths):

        """
        Runs the forward pass of the model.
        :param inputs: word ids (tokens) of shape (batch_size, window_size)
        :param lengths: array of actual lengths (no padding) of each input
        :return: the logits, a tensor of shape
                 (batch_size, window_size, vocab_size)
                 final_hidden, the final hidden state of the GRU in shape
                 (num_layers, batch, rnn_size)
        """
        # TODO: write forward propagation
        #print(inputs.size())
        x = self.embed(inputs)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths,
            batch_first=True, enforce_sorted=False)

        #x, final_hidden = self.gru(x)
        x, final_states = self.lstm(x)
        #print(lengths)
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
