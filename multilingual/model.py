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
        self.hidden_sz_1 = 64
        self.hidden_sz_2 = 256
        self.output_size = output_size
        self.bpe = bpe
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        #Pad idx???
        self.bpe_embed = nn.Embedding(self.vocab_size, self.embedding_size, 0)
        #for vanilla vocab
        self.enc_embed = nn.Embedding(self.vocab_size, self.embedding_size, 0)
        self.dec_embed = nn.Embedding(self.output_size, self.embedding_size, 0)

        self.encode_gru = nn.GRU(self.embedding_size, self.rnn_size, batch_first=True, bidirectional=True)
        #input size would be just embedding size if not concatenating encoder output
        #with decoder in
        #self.decode_gru = nn.GRU(self.embedding_size, dec_seq_len, batch_first=True)
        #self.decode_gru = nn.GRU(self.embedding_size+self.rnn_size, dec_seq_len, batch_first=True)
        self.decode_gru = nn.GRU(self.embedding_size+self.rnn_size*2, dec_seq_len, batch_first=True)
        #self.decode_gru = nn.GRU(self.embedding_size, self.rnn_size*2, batch_first=True)
        self.align = nn.Linear(2*self.rnn_size, self.rnn_size)
        self.fc1 = nn.Linear(self.rnn_size, self.hidden_sz_1)
        self.fc2 = nn.Linear(self.hidden_sz_1, self.hidden_sz_2)
        self.fc3 = nn.Linear(self.hidden_sz_2, self.output_size)
        self.fc3_bpe = nn.Linear(self.hidden_sz_2, self.vocab_size)

        # TODO: initialize embeddings, LSTM, and linear layers

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
        # TODO: write forward propagation

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
        if self.bpe:
            encode = self.bpe_embed(encoder_inputs)
            #encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)
            #we don't pack decode input yet since it will be concatenated to the encoder output (unless BROKEN!)
            decode = self.bpe_embed(decoder_inputs)
            #decode = pack_padded_sequence(decode, decoder_lengths, batch_first=True, enforce_sorted=False)

        else:
            encode = self.enc_embed(encoder_inputs)
            #encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)
            decode = self.dec_embed(decoder_inputs)
            #decode = pack_padded_sequence(decode, decoder_lengths, batch_first=True, enforce_sorted=False)

        encode_out, enc_hidden = self.encode_gru(encode)
        #print(enc_hidden.size())
        #print(encode_out.size())
        enc_hidden = torch.reshape(enc_hidden, (1, enc_hidden.size()[1], -1))
        enc_hidden = torch.squeeze(enc_hidden)
        #print(enc_hidden.size())
        context = F.softmax(self.align(enc_hidden), dim=1)
        #context = self.align(enc_hidden)
        context = torch.unsqueeze(context, 0)
        #print(context.size())

        context_decode = torch.cat((encode_out, decode), 2)
        #print(context_decode.size())
        #encode_out, encode_lengths = pad_packed_sequence(encode_out, batch_first=True)
        #if encode_out.size()[1] != self.rnn_size:
            #this occurs when all of the encoder sequences in the batch are < rnn_size
        #    encode_out = torch.cat((encode_out, torch.zeros(encode_out.size()[0], self.rnn_size - encode_out.size()[1], encode_out.size()[2])), 1)
        #decode_out, dec_hidden = self.decode_gru(decode, enc_hidden)
        decode_out, dec_hidden = self.decode_gru(context_decode, context)
        #decode_out, dec_hidden = self.decode_gru(context_decode, enc_hidden)
        #decode_out, decoder_lengths = pad_packed_sequence(decode_out, batch_first=True)

        decode_out = F.relu(self.fc1(decode_out))
        decode_out = F.relu(self.fc2(decode_out))
        if self.bpe:
            logits = self.fc3_bpe(decode_out)

        else:
            logits = self.fc3(decode_out)


        return logits

    def encode(self, encoder_inputs, encoder_lengths):
        if self.bpe:
            encode = self.bpe_embed(encoder_inputs)
            encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)

        else:
            encode = self.enc_embed(encoder_inputs)
            encode = pack_padded_sequence(encode, encoder_lengths, batch_first=True, enforce_sorted=False)

        #print(encoder_inputs.size())
        #print(decoder_inputs.size())
        encode_out, enc_hidden = self.encode_gru(encode)
        #print(pad_packed_sequence(encode_out, batch_first=True)[0].size())
        encode_out, _ = pad_packed_sequence(encode_out, batch_first=True)

        return encode_out, enc_hidden

    def decode(self, decoder_inputs, decoder_lengths, encoder_out, hidden):
        #we don't pack decode input yet since it will be concatenated to the encoder output
        decode_embed = self.bpe_embed(decoder_inputs) if self.bpe else self.dec_embed(decoder_inputs)
        #print(encoder_out.size())
        context_decode = torch.cat((encoder_out, decode_embed), 2)
        #print(context_decode.size())
        context_decode = pack_padded_sequence(context_decode, decoder_lengths, batch_first=True, enforce_sorted=False)
        #context_decode = pack_padded_sequence(decode_embed, decoder_lengths, batch_first=True, enforce_sorted=False)
        #UMMMMMMMMMMM, WHATS GOIN ON W/ UR DECODER INPUT?????
        decode_out, dec_hidden = self.decode_gru(context_decode, hidden)
        decode_out, _ = pad_packed_sequence(decode_out, batch_first=True)
        #print(decode_out.size())
        #print(self.fc1.size())

        decode_out = F.relu(self.fc1(decode_out))
        if self.bpe:
            logits = self.fc2_bpe(decode_out)

        else:
            logits = self.fc2(decode_out)

        return logits
