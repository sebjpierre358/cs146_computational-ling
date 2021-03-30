from comet_ml import Experiment
import torch
import torch.nn
import argparse
import math
import numpy as np
from preprocess import *
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": None,
     "num_epochs": None,
     "learning_rate": None,
     "window_size": None
 }


def train(model, train_loader, optimizer, experiment):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the training loop here, save trained model weights if needed
    model = model.train()
    with experiment.train():
        pass


def test(model, test_loader, experiment):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the testing loop and calculate perplexity
    model = model.eval()
    with experiment.validate():
        print("perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def interactive(input, tokenizer, model, top_k=10, ntok=20):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_l sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable token
    and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model
    response = None
    print(response)


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
    parser.add_argument("-i", "--interative", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed

    # Intialized the pretrained GPT-2 model and optimizer

    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer

    if args.load:
        model.load_state_dict(torch.load('model.pt'))
    if args.train:
        # run train loop here
        print("running training loop...")
        train(model, train_loader, optimizer, experiment)
    if args.save:
        torch.save(model.state_dict(), 'model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment)
    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, tokenizer, model)
