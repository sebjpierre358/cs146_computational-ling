from comet_ml import Experiment
from data import MyDataset, read_file
from model import BERT
from embedding_analysis import embedding_analysis
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "num_epochs":40,
    "batch_size": 216,
    "lr": .008,
    "win_sz": 50
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, loss_fn, optimizer, experiment):
    """
    Training loop that trains BERT model.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    """
    model = model.train()

    with experiment.train():
        print("Training {} epochs".format(hyperparams["num_epochs"]))
        for e in tqdm(range(hyperparams["num_epochs"])):
            #print("epoch {} / {}".format(e+1, hyperparams["num_epochs"]))
            for batch in train_loader:
            #for batch in tqdm(train_loader):
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()

                #of shape batch x win_sz x vocab_size
                logits = model(inputs)
                logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
        #perplexity = torch.exp(total_loss / total_predictions).item()

        #print("final epoch perplexity:", perplexity)


def test(model, test_loader, loss_fn, experiment):
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A BERT model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    """
    model.eval()
    total_loss = 0
    total_predictions = 0
    accuracy_sum = torch.zeros(1).to(device)
    num_batches = 0

    with experiment.test(), torch.no_grad():
        for batch in test_loader:
            num_batches += 1
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            total_predictions += torch.count_nonzero(labels).item()

            logits = model(inputs)
            logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

            accuracy_sum += compute_accuracy(logits, labels)
            total_loss += loss_fn(logits, labels)

        print(total_predictions)
        perplexity = torch.exp(total_loss / total_predictions).item()
        accuracy = (accuracy_sum[0] / num_batches).item()
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)

def compute_accuracy(logits, target):

    predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
    mask = target.ge(1)
    acc = torch.mean(torch.masked_select(torch.eq(predictions, target).type(torch.cuda.FloatTensor), mask))

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-a", "--analysis", action="store_true",
                        help="run embedding analysis")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    train_set = MyDataset(args.train_file, hyperparams['win_sz'])
    test_set = MyDataset(args.test_file, hyperparams['win_sz'], word2id=train_set.word2id)
    train_set.update_word2id(test_set.word2id)
    assert len(train_set) > len(test_set)

    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"],
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"],
                              pin_memory=True)

    vocab_size = test_set.vocab_size

    model = BERT(hyperparams["win_sz"], vocab_size, encoding='learned', n=6).to(device)
    #model = BERT(hyperparams["win_sz"], vocab_size, n=6).to(device)

    #assuming 0 = id for PAD
    loss_fn_trn = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_fn_tst = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    if args.load:
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        train(model, train_loader, loss_fn_trn, optimizer, experiment)
    if args.test:
        test(model, test_loader, loss_fn_tst, experiment)
    if args.save:
        torch.save(model.state_dict(), './model.pt')
    if args.analysis:
        embedding_analysis(model, experiment, train_set, test_set)
