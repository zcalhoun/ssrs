import os
import argparse
import logging
from itertools import chain

import torch
from torch.utils.data import DataLoader

from src import datasets, utils, metrics
from models import encoders, decoders

# Create parser
parser = argparse.ArgumentParser(description="Implementation of general encoder")

###################
# Data args
###################
parser.add_argument(
    "--task", type=str, default=None, help="The training task to attempt."
)
# TODO - set a reasonable default.
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for the data."
)

###################
# Encoder arguments
###################
parser.add_argument("--encoder", type=str, default="swav", help="The encoder to use.")

# Fine tuning for encoder
parser.add_argument(
    "--fine_tune_encoder", type=bool, default=False, help="Whether to fine tune or not."
)

###################
# Decoder arguments
###################
parser.add_argument("--decoder", type=str, default="unet", help="The decoder to use.")

###################
# Training arguments
###################
parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate.")
parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="Weight decay for parameters."
)
parser.add_argument(
    "--device", type=str, default="auto", help="Whether to use the GPU."
)
parser.add_argument(
    "--criterion", type=str, default="xent", help="Select the criterion to use."
)
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")


###################
# Logging arguments
###################
parser.add_argument(
    "--dump_path",
    type=str,
    default="./experiments/results/",
    help="Where to put the results for analysis.",
)
parser.add_argument(
    "--log_level", type=str, default="INFO", help="The log level to report on."
)


def main():
    # Set up arguments
    global args
    args = parser.parse_args()
    validate_args(args)

    # Set up timer to time results
    overall_timer = utils.Timer()

    # Set up path for recording results
    os.makedirs(args.dump_path)

    # Set up logger and log the arguments
    set_up_logger()
    logging.info(args)

    # Load the train dataset and the test dataset
    logging.info("Loading dataset...")
    train_data, test_data = datasets.load(args.task)

    # Create the dataloader
    logging.info("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    logging.info("Instantiating the model...")
    encoder = encoders.load(args.encoder)
    decoder = decoders.load(args.decoder, encoder)

    # Load model to GPU
    logging.info("Loading model to device...")
    device = set_device()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Set up optimizer, depending on whether
    # we are fine-tuning or not
    if args.fine_tune_encoder:
        # Chain the iterators to combine them.
        params = chain(encoder.parameters(), decoder.parameters())
    else:
        params = decoder.parameters()

    logging.info("Setting up optimizer and criterion...")
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = metrics.load(args.criterion, device)

    epoch_timer = utils.Timer()
    monitor = utils.PerformanceMonitor(args.dump_path)
    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"Beginning epoch {epoch}...")

        loss = train(train_loader, encoder, decoder, optimizer, criterion)
        monitor.log(epoch, "train", loss)

        loss = test(test_loader, encoder, decoder, criterion)
        monitor.log(epoch, "val", loss)
        logging.info(f"Epoch {epoch} took {epoch_timer.minutes_elapsed()} minutes.")
        epoch_timer.reset()

        if (epoch + 1) % 10 == 0:
            if loss < best_test_loss:
                logging.info("Saving model")
                save_model(encoder, decoder, args.dump_path, "best.pt")
                best_test_loss = loss

    logging.info(f"Code completed in {overall_timer.minutes_elapsed()}.")


def save_model(enc, dec, dump_path, name):
    torch.save(enc.state_dict(), os.path.join(dump_path, "enc_" + name))
    torch.save(dec.state_dict(), os.path.join(dump_path, "dec_" + name))


def train(loader, encoder, decoder, optimizer, criterion):

    if args.fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()

    decoder.train()
    criterion = criterion.cuda()
    avg_loss = utils.AverageMeter()
    for batch_idx, (inp, target) in enumerate(loader):
        logging.debug(f"Training batch {batch_idx}...")
        # Move to the GPU
        inp = inp.cuda()
        target = target.cuda()

        if args.fine_tune_encoder:
            output = encoder(inp)
        else:
            with torch.no_grad():
                output = encoder(inp)

        output = decoder(output)

        loss = criterion(output, target)

        # Calculate the gradients
        optimizer.zero_grad()
        loss.backward()
        avg_loss.update(loss.item(), inp.size(0))
        # Step forward
        optimizer.step()

    return avg_loss


@torch.no_grad()
def test(data_loader, encoder, decoder, criterion):

    encoder.eval()
    decoder.eval()
    criterion = criterion.cuda()
    avg_loss = utils.AverageMeter()
    for batch_idx, (inp, target) in enumerate(data_loader):
        # Move to the GPU
        inp = inp.cuda()
        target = target.cuda()

        # Compute output
        output = decoder(encoder(inp))
        loss = criterion(output, target)
        avg_loss.update(loss.item(), inp.size(0))

    return avg_loss


def validate_args():
    """
    This function ensures that several criteria
    are met before proceeding.
    """

    if args.task is None:
        raise Exception("A task must be specified.")
    if args.encoder is None:
        raise Exception("An encoder must be specified.")
    if args.decoder is None:
        raise Exception("A decoder must be specified.")


def set_up_logger():
    logging.basicConfig(
        filename=os.path.join(args.dump_path, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_device():
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    return device


if __name__ == "__main__":
    main()