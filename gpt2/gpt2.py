import torch
import torch.nn
import argparse
from transformers import *
from gpt2 import *
from transformer import *
from preprocess import *
from tqdm import tqdm
from comet_ml import Experiment

hyper_params = {
     "batch_size": 100,
     "num_epochs": 3,
     "learning_rate": 0.01
 }

experiment = Experiment(project_name="transformer")
experiment.log_parameters(hyper_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="transformer or gpt2")
    parser.add_argument("-bs", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # Load the GPT2 Tokenizer
   
    # Load the train, test DataLoader NOTE: Parse the data using the GPT2 tokenizer for both models

    if args.model == "transformer":
        # Load your transformer
        pass
    elif args.model == "gpt2":
        # Load the GPT2 model
        pass

    # Train the model if args.model == "transformer"


    # Test the model on the test set - report perplexity
    # NOTE: For the gpt2 model, you need to feed in one token at a time.
