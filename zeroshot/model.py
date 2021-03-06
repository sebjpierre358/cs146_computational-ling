from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, output_size,
                 enc_seq_len, dec_seq_len, bpe):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the input
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        :param output_size: The vocab size in output sequence
        :param enc_seq_len: The sequence length of encoder
        :param dec_seq_len: The sequence length of decoder
        :param bpe: whether the data is Byte Pair Encoded (shares same vocab)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.gru_layers=2
        self.hidden_sz = 256
        self.output_size = output_size
        self.bpe = bpe

        self.bpe_embed = nn.Embedding(self.vocab_size, self.embedding_size, 0)
        #for vanilla vocab
        self.enc_embed = nn.Embedding(self.vocab_size, self.embedding_size, 0)
        self.dec_embed = nn.Embedding(self.output_size, self.embedding_size, 0)

        self.encode_gru = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True, bidirectional=True,
                                 num_layers=self.gru_layers)
        #input size would be just embedding size if not concatenating encoder output
        self.decode_gru = nn.GRU(self.embedding_size+self.rnn_size*2, dec_seq_len, batch_first=True)

        #weights each of the encoder hidden states to each word in decoder sequence
        self.align = nn.Linear(2*self.rnn_size*self.gru_layers, self.rnn_size)

        self.fc1 = nn.Linear(self.rnn_size, self.hidden_sz)
        self.fc2 = nn.Linear(self.hidden_sz, self.output_size)

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths,
                decoder_lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """
        if self.bpe:
            encode = self.bpe_embed(encoder_inputs)
            #encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)
            decode = self.bpe_embed(decoder_inputs)
            #decode = pack_padded_sequence(decode, decoder_lengths, batch_first=True, enforce_sorted=False)

        else:
            encode = self.enc_embed(encoder_inputs)
            #encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)
            decode = self.dec_embed(decoder_inputs)
            #decode = pack_padded_sequence(decode, decoder_lengths, batch_first=True, enforce_sorted=False)

        encode_out, enc_hidden = self.encode_gru(encode)

        enc_hidden = torch.reshape(enc_hidden, (1, enc_hidden.size()[1], -1))
        enc_hidden = torch.squeeze(enc_hidden)

        context = F.softmax(self.align(enc_hidden), dim=1)
        context = torch.unsqueeze(context, 0)
        context_decode = torch.cat((encode_out, decode), 2)

        decode_out, dec_hidden = self.decode_gru(context_decode, context)

        decode_out = F.relu(self.fc1(decode_out))
        logits = self.fc2(decode_out)

        return logits
