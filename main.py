import argparse

import torch
from torch.utils.data import DataLoader

from src import datasets, utils
from decoders.models import load_decoder
from encoders.models import load_encoder

# Create parser
parser = argparse.ArgumentParser(description="Implementation of general encoder")

###################
# Data args
###################
parser.add_argument(
    "--task", type=str, default=None, help="The training task to attempt."
)
parser.add_argument("--batch_size", type=int, help="Batch size for the data.")


###################
# Encoder arguments
###################
parser.add_argument("--encoder", type=str, default=None, help="The encoder to use.")
parser.add_argument(
    "--fine_tune", type=bool, default=False, help="Whether to fine tune or not."
)

###################
# Decoder arguments
###################
parser.add_argument("--decoder", type=str, default=None, help="The decoder to use.")

###################
# Training arguments
###################
parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate.")
parser.add_argument(
    "--device", type=str, default="auto", help="Whether to use the GPU."
)


def main():
    global args
    args = parser.parse_args()

    if args.task is None:
        raise Exception("A task must be provided.")
    if args.encoder is None:
        raise Exception("An encoder must be provided.")
    if args.decoder is None:
        raise Exception("A decoder must be provided.")

    # Load the train dataset and the test dataset
    train_data, test_data = datasets.load(args.task)

    # Create the dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Instantiate the model
    encoder = load_encoder(args.encoder)
    decoder = load_decoder(args.decoder)

    # Load model to GPU
    if device == "auto":
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        self.device = args.device

    # Set up optimizer, depending on whether
    # we are fine-tuning or not
    if args.fine_tune:
        params = [encoder.paramaters(), decoder.parameters()]
    else:
        params = decoder.parameters()

    optimizer = utils.get_optim(args.task, params, args.lr)
    criterion = utils.get_criterion(args.task)

    for epoch in range(epochs):

        scores = train(train_loader, encoder, decoder, optimizer, criterion)

        scores = test(test_loader, encoder, decoder, criterion)


def train(loader, encoder, decoder, optimizer, criterion):

    if args.fine_tune:
        encoder.train()
    else:
        encoder.eval()

    decoder.train()
    criterion = criterion.cuda()

    for batch_idx, (inp, target) in enumerate(loader):
        # Move to the GPU
        inp = inp.cuda()
        target = target.cuda()

        if args.fine_tune:
            output = encoder(inp)
        else:
            with torch.no_grad():
                output = encoder(inp)

        output = decoder(output)

        loss = criterion(output, target)

        # Calculate the gradients
        optimizer.zero_grad()
        loss.backward()

        # Step forward
        optimizer.step()


@torch.no_grad()
def test(data_loader, encoder, decoder, criterion):

    encoder.eval()
    decoder.eval()
    criterion = criterion.cuda()

    for batch_idx, (inp, target) in enumerate(loader):
        # Move to the GPU
        inp = inp.cuda()
        target = target.cuda()

        # Compute output
        output = decoder(encoder(inp))
        loss = criterion(output, target)


if __name__ == "__main__":
    main()

