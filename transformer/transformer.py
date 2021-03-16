from comet_ml import Experiment
from model import Transformer
from preprocess import *
import argparse

hyperparams = {
    "batch_size": 128,
    "num_epochs": 3,
    "learning_rate": 0.01,
    "window_size": 30,
    "embedding_size": 100,
    "encoding_layers": 1,
    "attn_heads": 5,
    "enc_type": 'pos'
}

experiment = Experiment(project_name="transformer", log_code=False)
experiment.log_parameters(hyperparams)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

# Train the Model
def train(model, train_loader, experiment):
    total_loss = 0
    word_count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model = model.train()

    with experiment.train():
        for e in range(hyperparams["num_epochs"]):
            print("epoch {} / {}".format(e+1, hyperparams["num_epochs"]))
            for batch in tqdm(train_loader):
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()

                if e == hyperparams["num_epochs"]-1:
                    #assuming pad id is 0
                    word_count += torch.count_nonzero(inputs).item()

                #of shape batch x win_sz x vocab_size
                logits = model(inputs)
                logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

                loss = loss_fn(logits, labels)
                if e == hyperparams["num_epochs"]-1:
                    total_loss += loss
                #print(loss)
                loss.backward()
                optimizer.step()

        perplexity = torch.exp(total_loss / word_count).item()

        # Log perplexity to Comet.ml using experiment.log_metric
        print("final epoch perplexity:", perplexity)
        #print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        #experiment.log_metric("accuracy", accuracy)


# Test the Model
def test(model, test_loader, experiment):
    total_loss = 0
    word_count = 0

    with experiment.test():
        model = model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)

                word_count += torch.count_nonzero(inputs).item()

                logits = model(inputs)
                logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

                total_loss += loss_fn(logits, labels)

        perplexity = torch.exp(total_loss / word_count).item()

        print("test perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


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

    # Data Loader (Input Pipeline)
    train_loader, test_loader, vocab_size = load_dataset(args.train_file, args.test_file, hyperparams["window_size"],
                                             hyperparams["batch_size"])

    model = Transformer(vocab_size,
                        hyperparams["embedding_size"],
                        hyperparams["window_size"],
                        hyperparams["enc_type"],
                        hyperparams["encoding_layers"],
                        hyperparams["attn_heads"]).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
