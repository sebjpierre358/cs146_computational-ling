from comet_ml import Experiment
import torch
import torch.nn
import argparse
from transformers import *
from gpt2 import *
from transformer import *
from preprocess import *
from model import GPT2_Transformer
from tqdm import tqdm

hyperparams = {
     "num_epochs": 3,
     "learning_rate": 0.01,
     "embed_sz" : 768,
     "win_sz" : 35,
     "pos_enc" : 'learned',
     "heads" : 12,
 }

experiment = Experiment(project_name="gpt2", log_code=False)
experiment.log_parameters(hyperparams)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)

# Train the Model
def train(model, train_loader, experiment, is_gpt=False):
    total_loss = 0
    word_count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model = model.train()

    with experiment.train():
        for e in range(hyperparams["num_epochs"]):
            print("epoch {} / {}".format(e+1, hyperparams["num_epochs"]))
            for batch in tqdm(train_loader):
                labels = batch["labels"].to(device)
                if not is_gpt:
                    input = batch["inputs"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    if e == hyperparams["num_epochs"]-1:
                        word_count += torch.count_nonzero(input).item()

                    #of shape batch x win_sz x vocab_size
                    logits = model(input)
                    logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

                    loss = loss_fn(logits, labels)
                else:
                    input = batch["gpt_inputs"].to(device)
                    masks = batch["masks"].to(device)

                    optimizer.zero_grad()
                    num_tokens = torch.count_nonzero(input).item() - input.shape[0]
                    if e == hyperparams["num_epochs"]-1:
                        word_count += num_tokens

                    loss, logits = model(input, masks)
                    loss = loss * num_tokens

                total_loss += loss

                loss.backward()
                optimizer.step()

        perplexity = torch.exp(total_loss / word_count).item()

        print("final epoch perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


def test(model, test_loader, experiment, is_gpt=False):
    total_loss = 0
    word_count = 0

    with experiment.test():
        model = model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                labels = batch["labels"].to(device)
                if not is_gpt:
                    input = batch["inputs"].to(device)

                    word_count += torch.count_nonzero(input).item()

                    logits = model(input)
                    logits = torch.reshape(logits, (logits.size()[0], model.vocab_size, -1))

                    loss = loss_fn(logits, labels)
                else:
                    input = batch["gpt_inputs"].to(device)
                    mask = batch["mask"].to(device)
                    num_tokens = torch.count_nonzero(input).item() - input.shape[0]
                    word_count += num_tokens

                    loss, logits = model(input, mask)
                    loss = loss * num_tokens

                total_loss += loss

        perplexity = torch.exp(total_loss / word_count).item()

        print("test perplexity:", perplexity)
        experiment.log_metric("perplexity", perplexity)


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

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    train_loader, test_loader = load_dataset(args.train_file, args.test_file, tokenizer, args.batch_size,
                                             hyperparams["win_sz"])


    if args.model == "transformer":
        model = Transformer(embed_size=hyperparams['embed_sz'],
                            window_size=hyperparams['win_sz'],
                            encoding=hyperparams['pos_enc'],
                            heads=hyperparams['heads'],
                            ff_dim=4*hyperparams['embed_sz']).to(device)

    elif args.model == "gpt2":
        gpt_config = GPT2Config(n_positions=hyperparams['win_sz'],
                                n_ctx=hyperparams['win_sz'],
                                n_embed=hyperparams['embed_sz'],
                                n_head=hyperparams['heads'],
                                output_hidden_states=False,
                                output_attentions=False,
                                use_cache=False,
                                return_dict=False)


        model = GPT2_Transformer(gpt_config).to(device)

    if args.train:
        print("running training loop...")
        if args.model=='gpt2':
            train(model, train_loader, experiment, True)
        elif args.model=='transformer':
            train(model, train_loader, experiment)
    if args.test:
        print("running testing loop...")
        if args.model=='gpt2':
            test(model, test_loader, experiment, True)
        elif args.model=='transformer':
            test(model, test_loader, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
