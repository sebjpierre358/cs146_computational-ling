import sys

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from model import BERT
from data import MyDataset


def plot_embeddings(texts, embeddings, plot_name):
    """
    Uses MDS to plot embeddings (and its respective sentence) in 2D space.

    Inputs:
    - texts: A list of strings, representing the words
    - embeddings: A 2D numpy array, [num_sentences x embedding_size],
        representing the relevant word's embedding for each sentence
    """
    embeddings = embeddings.astype(np.float64)
    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(embeddings)

    plt.figure(1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color='navy')
    for i, text in enumerate(texts):
        plt.annotate(text, (embeddings[i, 0], embeddings[i, 1]))
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(plot_name, dpi=100)


def embedding_analysis(model, experiment, train_set, test_set):
    """
    Create embedding analysis image for each list of polysemous words and
    upload them to comet.ml.

    Inputs:
    - model: Trained BERT model
    - experiment: comet.ml experiment object
    - train_set: train dataset
    - test_set: test dataset
    """
    polysemous_words = {
        "figure": ["figure", "figured", "figures"],
        "state": ["state", "states", "stated"],
        "bank": ["bank", "banks", "banked"]
    }

    for key in polysemous_words:
        # TODO: Find all instances of sentences that have polysemous words.

        # TODO: Give these sentences as input, and obtain the specific word
        #       embedding as output.

        # TODO: Use the plot_embeddings function above to plot the sentence
        #       and embeddings in two-dimensional space.

        # TODO: Save the plot as "{word}.png"
        experiment.log_image(f"{key}.png")
