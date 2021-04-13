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
    print("plotting {} with {} examples".format(plot_name, len(texts)))
    #print(len(texts))
    embeddings = embeddings.astype(np.float64)
    mds = MDS(n_components=2)
    #print(embeddings.size)
    embeddings = mds.fit_transform(embeddings)

    plt.figure(1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color='navy')
    for i, text in enumerate(texts):
        #print(text)
        plt.annotate(text, (embeddings[i, 0], embeddings[i, 1]))
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(plot_name, dpi=100)
    fig.clear()
    #print("------------------------------------------------")


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
        "show" : ["show", "shows", "showing"],
        "figure": ["figure", "figured", "figures"],
        "state": ["state", "states", "stated"],
        "bank": ["bank", "banks"]
    }

    for key in polysemous_words:
        word_list = polysemous_words[key]
        # TODO: Find all instances of sentences that have polysemous words.
        train_instances, train_indices, trn_words = train_set.get_instances(word_list)
        test_instances, test_indices, tst_words = test_set.get_instances(word_list)
        instances = torch.cat((train_instances, test_instances))
        indices = np.concatenate((train_indices, test_indices))
        word_order = trn_words + tst_words

        #print(indices.shape)
        #print(instances.shape)

        # TODO: Give these sentences as input, and obtain the specific word
        #       embedding as output.
        full_embeds = model.get_embeddings(instances).cpu()
        full_embeds = full_embeds.detach().numpy()
        embeds = []
        for i in range(len(full_embeds)):
            embeds.append(full_embeds[i][indices[i]])
        #print(len(embeds))
        embeds = np.stack(embeds)

        #print(embeds.shape)

        #embeds = np.choose(indices, embeds)
        #embeds = np.take(embeds, indices, 1)
        #print(embeds.shape)
        #check shape

        plot_embeddings(word_order, embeds, f"{key}.png")

        # TODO: Use the plot_embeddings function above to plot the sentence
        #       and embeddings in two-dimensional space.

        # TODO: Save the plot as "{word}.png"
        experiment.log_image(f"{key}.png")
