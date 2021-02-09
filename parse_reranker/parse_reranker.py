from comet_ml import Experiment
from preprocess import ParsingDataset, RerankingDataset
from model import LSTMLM
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
import math
from tqdm import tqdm  # optional progress bar
from torch.utils.data.sampler import SubsetRandomSampler

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 30,
    "embedding_size": 50,
    "num_epochs": 1,
    "batch_size": 64,
    "learning_rate": .001
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # TODO: Write training loop
    model = model.train()
    with experiment.train():
        for batch in tqdm(train_loader):
            inputs = batch["parse"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            logits = model(inputs, batch["length"])
            #print(logits.size())
            #print(labels.size())

            logits = torch.reshape(logits, (hyperparams["batch_size"],
                                   model.vocab_size, hyperparams["rnn_size"]))

            loss = loss_fn(logits, labels)
            #print(loss)
            loss.backward()
            optimizer.step()


def validate(model, validate_loader, experiment, hyperparams):
    """
    Validates the model performance as LM on never-seen data.
    :param model: the trained model to use for prediction
    :param validate_loader: Dataloader of validation data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    """
    # TODO: Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_idx)
    total_loss = 0
    word_count = 0

    # TODO: Write validating loop
    model = model.eval()
    with experiment.validate():
        with torch.no_grad():
            for batch in validate_loader:
                inputs = batch["parse"].to(device)
                labels = batch["label"].to(device)
                word_count += torch.sum(batch["length"])
                logits = model(inputs, batch["length"])
                logits = torch.reshape(logits, (hyperparams["batch_size"],
                         model.vocab_size, hyperparams["rnn_size"]))

                total_loss += loss_fn(logits, labels)

        perplexity = math.exp(total_loss / word_count)
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def test(model, test_dataset, experiment, hyperparams):
    """
    Validates and tests the model for parse reranking.
    :param model: the trained model to use for prediction
    :param test_dataset: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: Hyperparameters dictionary
    """
    # TODO: Write testing loops
    model = model.eval()
    with experiment.test():

        precisions = []
        recalls = []

        with torch.no_grad():
            for s in tqdm(range(test_dataset.__len__())):
                input_batch = test_dataset.__getitem__(s)
                gold_constits = input_batch["gold"]
                #print(input_batch["trees"])
                num_correct = input_batch["trees"][:, 0]
                total_constits = input_batch["trees"][:, 1]
                lengths = input_batch["trees"][:, 2]
                last_tokens = input_batch["trees"][:, 3]
                trees = input_batch["trees"][:, 4:].to(device)

                prbs = model(trees, lengths)
                #use lengths.size() as substitute for # of trees for the given sentence
                prbs = F.softmax(torch.reshape(prbs, (lengths.size()[0],
                                     model.vocab_size, -1)), dim=1) #-1 here is rnn_size

                sentence_prbs = []
                for t in range(last_tokens.size()[0]):
                    sentence_prbs.append(prbs[t][last_tokens[t]][-1])

                best_idx = np.argmax(sentence_prbs)
                precisions.append((num_correct[best_idx] / total_constits[best_idx]))
                recalls.append((num_correct[best_idx] / gold_constits))

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = (2 * precision * recall) / (precision + recall)
        print("precision:", precision)
        print("recall:", recall)
        print("F1:", f1)
        experiment.log_metric("precision", precision)
        experiment.log_metric("recall", recall)
        experiment.log_metric("F1", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("parse_file")
    parser.add_argument("gold_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="run validation loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    dataset = ParsingDataset("data/reranker_train.txt")
    hyperparams["rnn_size"] = dataset.avg_len
    experiment.log_parameters(hyperparams)

    train_sampler = SubsetRandomSampler(dataset.train_indices)
    validate_sampler = SubsetRandomSampler(dataset.validate_indices)

    train_loader = DataLoader(dataset, batch_size=hyperparams["batch_size"],
                              shuffle=False, drop_last=True, pin_memory=True,
                              sampler=train_sampler)

    validate_loader = DataLoader(dataset, batch_size=hyperparams["batch_size"],
                              shuffle=False, drop_last=True, pin_memory=True,
                              sampler=validate_sampler)

    test_dataset = RerankingDataset("data/conv.txt", "data/gold.txt", dataset.word2id, hyperparams["rnn_size"])

    model = LSTMLM(
        dataset.vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        dataset.pad_idx
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams)
    if args.validate:
        print("running validation...")
        validate(model, validate_loader, experiment, hyperparams)
    if args.test:
        print("testing reranker...")
        test(model, test_dataset, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
