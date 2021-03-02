from comet_ml import Experiment
from preprocess import preprocess_vanilla, TranslationDataset
from model import Seq2Seq
from model_tf import RNN_Seq2Seq
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm  # optional progress bar

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 30,  # assuming encoder and decoder use the same rnn_size
    "embedding_size": 100,
    "num_epochs": 6,
    "batch_size": 64,
    "learning_rate": .002
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams, bpe, is_tf):
    """
    Trains the model.

    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) #the pad id

    if not is_tf:
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
                    #print(loss)
                    loss.backward()
                    optimizer.step()

    else:
        with experiment.train():
            for e in range(hyperparams["num_epochs"]):
                print("epoch {} / {}".format(e+1, hyperparams["num_epochs"]))
                for batch in tqdm(train_loader):
                    enc_input = torch2flow(batch["enc_input"])
                    dec_input = torch2flow(batch["dec_input"])
                    target = batch["labels"]
                    mask = torch2flow(target.ge(1).type(torch.LongTensor))
                    target = torch2flow(target)

                    with tf.GradientTape() as tape:
                        prbs = model.call(enc_input, dec_input)
                        batch_loss = model.loss_function(prbs, target, mask)

                    gradients = tape.gradient(batch_loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_loader, experiment, hyperparams, bpe, is_tf):
    """
    Validates the model performance as LM on never-seen data.

    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
    total_loss = 0
    loss_sum = tf.constant([0.0], dtype=tf.float32)
    word_count = 0
    num_batches = 0
    accuracy_sum = tf.constant([0.0], dtype=tf.float32) if is_tf else torch.zeros(1).to(device)

    with experiment.test():
        if not is_tf:
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

        else:
            for batch in test_loader:
                num_batches += 1
                enc_input = torch2flow(batch["enc_input"])
                dec_input = torch2flow(batch["dec_input"])
                target = batch["labels"]
                mask = torch2flow(target.ge(1).type(torch.LongTensor))
                target = torch2flow(target)

                prbs = model.call(enc_input, dec_input)
                batch_loss = model.loss_function(prbs, target, mask)
                loss_sum = tf.add(loss_sum, batch_loss)
                batch_accuracy = model.accuracy_function(prbs, target, mask)
                accuracy_sum = tf.add(accuracy_sum, batch_accuracy)
                word_count += torch.sum(batch["dec_len"]).numpy()

            total_avg_loss = np.divide(loss_sum.numpy(), np.asarray([word_count]))
            accuracy = np.around(np.divide(accuracy_sum.numpy(), num_batches), decimals=3)
            perplexity = np.exp(total_avg_loss)

        print("ok, we made it")
        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)


def torch2flow(tensor):
    return tf.convert_to_tensor(tensor.numpy())

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
    args = parser.parse_args()
    assert len(args.corpus_files) == len(args.multilingual_tags)

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    if args.bpe:
        dataset = TranslationDataset(args.corpus_files[0], hyperparams["rnn_size"],
                  hyperparams["rnn_size"])

    else:
        dataset = TranslationDataset(args.corpus_files[0], hyperparams["rnn_size"],
                  hyperparams["rnn_size"], bpe=False)

    vocab_size = dataset.vocab_size
    output_size = dataset.output_size

    train_sz = int(len(dataset) * .9)
    validate_sz = len(dataset) - train_sz
    train_set, test_set = random_split(dataset, [train_sz, validate_sz])
    train_loader = DataLoader(train_set, batch_size=hyperparams["batch_size"],
                              shuffle=True, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"],
                              shuffle=True, pin_memory=True)

    # Hint: Use ConcatDataset to concatenate datasets
    # Hint: Make sure encoding and decoding lengths match for the datasets
    data_tags = list(zip(args.corpus_files, args.multilingual_tags))

    model = Seq2Seq(
        vocab_size,
        hyperparams["rnn_size"],
        hyperparams["embedding_size"],
        output_size,
        hyperparams["rnn_size"], #enc_seq_len
        hyperparams["rnn_size"], #dec_seq_len
        args.bpe
    ).to(device)

    model_tf = RNN_Seq2Seq(
        hyperparams["rnn_size"], #enc_seq_len
        vocab_size, #enc lang vocab
        hyperparams["rnn_size"],
        output_size,
        args.bpe)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model_tf, train_loader, experiment, hyperparams, args.bpe, True)
        #train(model, train_loader, experiment, hyperparams, args.bpe, False)
    if args.test:
        print("running testing loop...")
        test(model_tf, test_loader, experiment, hyperparams, args.bpe, True)
        #test(model, test_loader, experiment, hyperparams, args.bpe, False)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
