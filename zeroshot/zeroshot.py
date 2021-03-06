from comet_ml import Experiment
from preprocess import TranslationDataset, ZeroshotDataset
from model import Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 40,  # assuming encoder and decoder use the same rnn_size
    "embedding_size": 128,
    "num_epochs":8,
    "batch_size": 128,
    "learning_rate": .003
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams, bpe):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is dataset bpe or not
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) #the pad id
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model = model.train()
    with experiment.train():
        for e in range(hyperparams["num_epochs"]):
            print("epoch {} / {}".format(e+1, hyperparams["num_epochs"]))
            for batch in tqdm(train_loader):
                enc_input = batch["enc_input"].to(device)
                dec_input = batch["dec_input"].to(device)
                target = batch["labels"].to(device)
                optimizer.zero_grad()

                #of shape batch x seq_len x vocab_size
                logits = model(enc_input, dec_input, batch["enc_len"], batch["dec_len"])
                logits = torch.reshape(logits, (logits.size()[0], model.output_size, -1))
                #when batch sequences true lengths all < rnn size
                # necessary to have equal dims for loss
                if logits.size()[2] != target.size()[0]:
                    target = target[:, :logits.size()[2]]

                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()


def test(model, test_loader, experiment, hyperparams, bpe):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is dataset bpe or not
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    total_loss = 0
    word_count = 0
    num_batches = 0
    accuracy_sum = torch.zeros(1).to(device)

    model = model.eval()
    with experiment.test():
        model = model.eval()
        with torch.no_grad():
            for batch in test_loader:
                num_batches += 1
                enc_input = batch["enc_input"].to(device)
                dec_input = batch["dec_input"].to(device)
                target = batch["labels"].to(device)

                word_count += torch.sum(batch["dec_len"])

                #of shape batch x seq_len x vocab_size
                logits = model(enc_input, dec_input, batch["enc_len"], batch["dec_len"])
                logits = torch.reshape(logits, (logits.size()[0], model.output_size, -1))
                if logits.size()[2] != target.size()[0]:
                    target = target[:, :logits.size()[2]]

                accuracy_sum += compute_accuracy(logits, target)
                total_loss += loss_fn(logits, target)

        perplexity = torch.exp(total_loss / word_count).item()
        accuracy = (accuracy_sum[0] / num_batches).item()

        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)

def compute_accuracy(logits, target):

    predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
    mask = target.ge(1) #skip the padding idx (0) for accuracy calc
    acc = torch.mean(torch.masked_select(torch.eq(predictions, target).type(torch.cuda.FloatTensor), mask))

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--corpus-files", nargs="*")
    parser.add_argument("-b", "--bpe", action="store_true",
                        help="use bpe data")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-m", "--multilingual-tags", nargs="*", default=[None],
                        help="target tags for translation")
    parser.add_argument("-z", "--zeroshot", action="store_true",
                        help="zeroshot translation")
    args = parser.parse_args()
    args_valid = True

    # TODO: Make sure you modify the `.comet.config` file
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)


    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    # Hint: Use ConcatDataset to concatenate datasets
    # Hint: Make sure encoding and decoding lengths match for the datasets
    data_tags = list(zip(args.corpus_files, args.multilingual_tags))

    if not args.zeroshot:
        if len(data_tags) == 1:
            print("initiating one-to-one translation")
            dataset = TranslationDataset(args.corpus_files[0], hyperparams["rnn_size"],
                      hyperparams["rnn_size"], target=data_tags[0][1])

            vocab_size = dataset.vocab_size
            output_size = dataset.output_size

        elif len(data_tags) == 2:
            if (data_tags[0][1] == None or data_tags[1][1] == None):
                print("Need both tags for multilingual translation")
                args_valid = False
            else:
                print("initiating one-to-many translation")
                dataset_1 = TranslationDataset(data_tags[0][0], hyperparams["rnn_size"],
                    hyperparams["rnn_size"], target=data_tags[0][1])
                dataset_2 = TranslationDataset(data_tags[1][0], hyperparams["rnn_size"],
                    hyperparams["rnn_size"], target=data_tags[1][1], word2id=dataset_1.word2id)

                dataset_1.word2id = dataset_2.word2id
                vocab_size = dataset_2.vocab_size
                output_size = dataset_2.output_size

                dataset = ConcatDataset([dataset_1, dataset_2])

        train_sz = int(len(dataset) * .9)
        validate_sz = len(dataset) - train_sz
        train_set, test_set = random_split(dataset, [train_sz, validate_sz])

        train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"],
                                  shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"],
                                 shuffle=True, pin_memory=True)

    else:
        if len(data_tags) != 4:
            print("need 2 preprocessed test sets for lang X->Y and Y->Z in addition to the train sets")
            print("got {} files out of 4 needed".format(len(data_tags)))
            args_valid = False
        else:
            #expecting tag in format <2lang_name>
            source_tag = data_tags[2][1]
            target_tag = data_tags[3][1]
            print("initiating zeroshot translation to {}".format(target_tag[2:-1]))
            trainset_1 = TranslationDataset(data_tags[0][0], hyperparams["rnn_size"],
                hyperparams["rnn_size"], target=data_tags[0][1], flip=False)
            trainset_2 = TranslationDataset(data_tags[1][0], hyperparams["rnn_size"],
                hyperparams["rnn_size"], target=data_tags[1][1], word2id=trainset_1.word2id)

            #may not be necessary
            dataset = ConcatDataset([trainset_1, trainset_2])

            test_set = ZeroshotDataset(data_tags[2][0], data_tags[3][0], target_tag,
                trainset_2.word2id, hyperparams["rnn_size"])

            trainset_1.word2id = test_set.word2id
            trainset_2.word2id = test_set.word2id

            vocab_size = test_set.vocab_size
            output_size = test_set.output_size

            train_loader = DataLoader(dataset, batch_size=hyperparams["batch_size"],
                                      shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"],
                                      shuffle=True, pin_memory=True)

    if args_valid:

        model = Seq2Seq(
            vocab_size,
            hyperparams["rnn_size"],
            hyperparams["embedding_size"],
            output_size,
            hyperparams["rnn_size"],
            hyperparams["rnn_size"],
            args.bpe
        ).to(device)

        if args.load:
            print("loading saved model...")
            model.load_state_dict(torch.load('./model.pt'))
        if args.train:
            print("running training loop...")
            train(model, train_loader, experiment, hyperparams, args.bpe)
        if args.test:
            print("running testing loop...")
            test(model, test_loader, experiment, hyperparams, args.bpe)
        if args.save:
            print("saving model...")
            torch.save(model.state_dict(), './model.pt')
