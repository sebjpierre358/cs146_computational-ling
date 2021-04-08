from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class BERT(nn.Module):
    def __init__(self, seq_len, num_words, d_model=512, h=8, n=6):
        super().__init__()
        # TODO: Initialize BERT modules

    def forward(self, x):
        # TODO: Write feed-forward step
        pass

    def get_embeddings(self, x):
        # TODO: Write function that returns BERT embeddings of a sequence
        pass
