from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch

def load_dataset(fn, tokenizer, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
    So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
    the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
    """
    pass