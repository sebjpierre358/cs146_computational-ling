from comet_ml import Experiment
from preprocess import *
import argparse

hyper_params = {
    "batch_size": 100,
    "num_epochs": 3,
    "learning_rate": 0.01
}

experiment = Experiment(project_name="transformer")
experiment.log_parameters(hyper_params)

# Data Loader (Input Pipeline)
train_loader, test_loader = load_dataset()

# Initialize your transformer using the hyper-parameters

# Loss and Optimizer


# Train the Model
def train(experiment, ...):
    with experiment.train():
        pass
        # Forward + Backward + Optimize

        # Compute train accuracy

        # Log perplexity to Comet.ml using experiment.log_metric


# Test the Model
def test(experiment, ...):
    with experiment.test():
        pass
        # Log perplexity to Comet.ml using experiment.log_metric


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
    args = parser.parse_args()
