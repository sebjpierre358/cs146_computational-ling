from torch.utils.data import Dataset, DataLoader
import torch


def load_dataset(fn, tokenizer, batch_size, window_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for
    train and test

    :Comment: This function should be similary as the last GPT-2 assignemnent.
    We are still using the GPT-2 tokenizer to get the word ids.
    One thing to note is that when preprocessing the dataset, please exclude
    all the sentences defining the speaker's persona, in this assignment,
    we will just be training and testing on the chat text data. Also for each
    record of chat in the format of "sentence \t response", add BOS and EOS
    token to the start and end it, also add a special token between sentence
    and response to differentiate them from each other.
    """
    pass
