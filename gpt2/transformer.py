import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    def __init__(self, embed_size, window_size, encoding, heads=1, layers=1, ff_dim=256, vocab_size=50257):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.window_size = window_size
        self.encoding = encoding
        self.layers = layers
        if self.encoding == 'learned':
            self.pos_embed = PosEncoding(self.window_size, self.embed_size)

        if self.layers == 1:
            self.encode_block = TransEncoder(self.window_size, self.embed_size, heads, ff_dim)
        else:
            self.encode_block = nn.ModuleList([TransEncoder(
                self.window_size, self.embed_size, heads, ff_dim) for i in range(self.layers)])

        self.norm = nn.LayerNorm([self.window_size, self.embed_size])

        self.fwrd_out_1 = nn.Linear(self.embed_size, ff_dim)
        self.fwrd_out_2 = nn.Linear(ff_dim, self.vocab_size)

        self.embed = nn.Embedding(self.vocab_size, self.embed_size, 0)

    def forward(self, input):
        embeds = self.embed(input)
        if self.encoding == 'sinusoid':
            embeds = sinusoid_encoding(embeds)
        elif self.encoding == 'learned':
            embeds = self.pos_embed(embeds)

        if self.layers == 1:
            encode_out = self.encode_block(embeds)
        else:
            encode_out = embeds
            for encoder in self.encode_block:
                encode_out = self.norm(encode_out + encoder(encode_out))

        output = F.relu(self.fwrd_out_1(encode_out))
        output = self.fwrd_out_2(output)

        assert output.shape == (input.shape[0], self.window_size, self.vocab_size)

        return output

class TransEncoder(nn.Module):
    def __init__(self, window_size, embed_size, heads, ff_dim=256):
        super(TransEncoder, self).__init__()

        self.embed_size = embed_size
        self.window_size = window_size

        self.fc_1 = nn.Linear(self.embed_size, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, self.embed_size)
        self.norm = nn.LayerNorm([self.window_size, self.embed_size])

        if heads == 1:
            self.attn = SelfAttn(self.window_size, self.embed_size)
        else:
            #convenient if embed_size is divisible by num_heads
            self.attn = MultiHeadAttn(heads, embed_size, window_size)

    def forward(self, embeds):
        attn_out = self.attn(embeds)
        resid_1 = self.norm(attn_out + embeds)
        fc_out = F.relu(self.fc_1(resid_1))
        fc_out = self.fc_2(fc_out)
        encode_out = self.norm(resid_1 + fc_out)

        return encode_out


class SelfAttn(nn.Module):
    def __init__(self, window_sz, embed_sz):
        super(SelfAttn, self).__init__()
        self.embed_sz = embed_sz
        self.window_sz = window_sz

        self.W_q = nn.Linear(self.embed_sz, self.embed_sz)
        self.W_k = nn.Linear(self.embed_sz, self.embed_sz)
        self.W_v = nn.Linear(self.embed_sz, self.embed_sz)

        self.mask = torch.triu(torch.full((self.window_sz, self.window_sz), -math.inf), diagonal=1).to(device)
        #dim 1 is the window size
        self.soft = nn.Softmax(dim=1)

    def forward(self, embeddings):
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)
        assert Q.shape == K.shape == V.shape == (embeddings.shape[0], self.window_sz, self.embed_sz)

        A = self.attn(Q, K, V)
        assert A.shape == (embeddings.shape[0], self.window_sz, self.embed_sz)

        return A


    def attn(self, Q, K, V):
        #dim 0 is the batch axis, so don't want to transpose that
        A = torch.div(torch.matmul(Q, torch.transpose(K, 1, 2)), math.sqrt(self.embed_sz))

        A = A + self.mask
        A = torch.matmul(self.soft(A), V)

        return A

class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, embed_sz, window_sz):
        super(MultiHeadAttn, self).__init__()
        self.num_heads = num_heads
        self.embed_sz = embed_sz
        self.window_sz = window_sz
        self.sub_dim = int(embed_sz/self.num_heads)
        self.soft = nn.Softmax(dim=1)
        self.mask = torch.triu(torch.full((self.window_sz, self.window_sz), -math.inf), diagonal=1).to(device)

        #want to learn the projection of query, key, and val into new dimension
        #instead of reshaping
        self.project_queries = nn.ModuleList([nn.Linear(self.embed_sz, self.sub_dim) for i in range(self.num_heads)])
        self.project_keys = nn.ModuleList([nn.Linear(self.embed_sz, self.sub_dim) for j in range(self.num_heads)])
        self.project_vals = nn.ModuleList([nn.Linear(self.embed_sz, self.sub_dim) for k in range(self.num_heads)])
        self.project_full = nn.Linear(self.embed_sz, self.embed_sz)

    def forward(self, inputs):
        #shape will be num_heads*batch x window_size x sub_dim
        #this cat is to run attention calculation in parallel
        sub_queries = torch.cat([query_W(inputs) for query_W in self.project_queries])
        sub_keys = torch.cat([key_W(inputs) for key_W in self.project_keys])
        sub_vals = torch.cat([value_W(inputs) for value_W in self.project_vals])
        assert sub_vals.shape == (self.num_heads*inputs.shape[0], self.window_sz, self.sub_dim)

        #every batch_sz grouping is all the attentions for a single head
        batch_attns = self.attn(sub_queries, sub_keys, sub_vals)
        assert batch_attns.shape == (self.num_heads*inputs.shape[0], self.window_sz, self.sub_dim)
        #to return to full-sized batch x window_sz x embed_sz tensor
        x = torch.cat(torch.split(batch_attns, inputs.shape[0]), dim=2)
        assert x.shape == (inputs.shape[0], self.window_sz, self.embed_sz)

        x = self.project_full(x)
        return x


    def attn(self, Q, K, V):
        A = torch.div(torch.matmul(Q, torch.transpose(K, 1, 2)), math.sqrt(self.sub_dim))
        A = A + self.mask
        A = torch.matmul(self.soft(A), V)

        #in shape batch*num_heads x window_size x sub_dim
        return A

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
    #should be able to do the sin/cos all in a list comprehension with a conditional...
    enc = np.fromfunction(encode_angle, (window_sz, embed_sz))
    #takes all the even rows
    enc[::2] = np.sin(enc[::2])
    #all odds
    enc[1::2] = np.cos(enc[1::2])

    return input + torch.FloatTensor(enc).to(device)
