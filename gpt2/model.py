import torch
import torch.nn as nn
from transformers import *

class GPT2_Transformer(nn.Module):
    def __init__(self, config, vocab_size=50257):
        super(GPT2_Transformer, self).__init__()
        self.gpt_lm = GPT2LMHeadModel(config).from_pretrained('gpt2')
        self.vocab_size = vocab_size


        """
        :param input: Batched tensor of inputs for the model
        :param attn_mask: Tensor of 1s and 0s to prevent calculating attention
                          on pad tokens
        """
    def forward(self, input, attn_mask):

        #we use the inputs as labels for the gpt since it accounts for the shift
        lbls = torch.where(input!= 0.0, input, -100)
        loss, logits = self.gpt_lm(input, attention_mask=attn_mask, labels=lbls, output_hidden_states=False, return_dict=False, output_attentions=False, use_cache=False)

        return loss, logits
