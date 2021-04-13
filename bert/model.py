from torch import nn
from torch.nn import functional as F
import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT(nn.Module):
    def __init__(self, seq_len, vocab_size, encoding='sinusoid', d_model=512, h=8, n=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        ff_dim = 4 * d_model
        self.d_model = d_model
        #self.word2id = word2id

        self.encoding = encoding
        if self.encoding == 'learned':
            self.pos_embed = PosEncoding(self.seq_len, d_model)

        self.norm = nn.LayerNorm([self.seq_len, d_model])
        #assume 0 is the id for PAD
        self.embed = nn.Embedding(self.vocab_size, d_model, 0)

        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, h), n, self.norm)
        #self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, h), n)
        #self.fwrd_out_1 = nn.Linear(d_model, ff_dim)
        #self.fwrd_out_2 = nn.Linear(ff_dim, self.vocab_size)
        self.linear_out = nn.Linear(d_model, self.vocab_size)

    def forward(self, x):
        embeds = self.embed(x)
        if self.encoding == 'sinusoid':
            embeds = sinusoid_encoding(embeds)
        elif self.encoding == 'learned':
            embeds = self.pos_embed(embeds)

        enc_out = self.transformer_enc(embeds)

        #output = F.relu(self.fwrd_out_1(enc_out))
        #output = self.fwrd_out_2(output)
        output = self.linear_out(enc_out)

        assert output.shape == (x.shape[0], self.seq_len, self.vocab_size)

        return output

    def get_embeddings(self, x):
        # TODO: Write function that returns BERT embeddings of a sequence
        with torch.no_grad():
            x = self.embed(x.to(device))
            if self.encoding == 'sinusoid':
                x = sinusoid_encoding(x)
            elif self.encoding == 'learned':
                x = self.pos_embed(x)

            #x = self.transformer_enc(x)
            #assert x.shape == (x.shape[0], self.seq_len, self.d_model)

            return x

class PosEncoding(nn.Module):
    def __init__(self, window_sz, emb_sz):
        super(PosEncoding, self).__init__()
        self.pos_embed = torch.randn((window_sz, emb_sz), requires_grad=True).to(device)

    def forward(self, inputs):
        return self.pos_embed + inputs

def sinusoid_encoding(input):
    embed_sz = input.shape[2]
    window_sz = input.shape[1]

    def encode_angle(pos, i):
        return pos / (10000.0**(2.0*i/embed_sz))

    assert embed_sz % 2 == 0
    enc = np.fromfunction(encode_angle, (window_sz, embed_sz))
    #takes all the even rows
    enc[::2] = np.sin(enc[::2])
    #all odds
    enc[1::2] = np.cos(enc[1::2])

    return input + torch.FloatTensor(enc).to(device)
