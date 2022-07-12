"""
This code handles the analysis process so reports can easily be generated that
evaluate how well models perform.

Visuals to include:
* Training / validation curves
* A few training examples (todo -- look at which examples are particularly difficult)
* Precision / recall curves. (todo -- create this curve automatically)

Metrics to include:
* IoU of final model and of best model.
* Pixel based Precision / Recall.

"""

import os
import argparse
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import joblib

from PIL import Image

from src import datasets
from models import encoders, decoders

import torch
from torch.nn import functional as F


def main(args):

    # Check if directory exists. If not, then create the directory

    if ~os.path.exists(args.dump_path):
        print("Path doesn't exist")
        os.makedirs(args.dump_path)

    # First off, create the train/val curve.
    create_train_val_curve(args.experiment_path)

    # You only need the test dataset output.
    _, test_dataset = datasets.load(task=args.task, normalization=args.normalization)

    model = Model(args.experiment_path, args.device, args.model_type)
    model.to(args.device)

    # TODO -- add in the ability to calculate Precision Recall curves.
    prc = PrecisionRecall(np.linspace(0, 1, 20))
    calc_iou = JaccardIndex()
    iou_results = []
    # Iterate through the test images.
    for i, (img, mask) in enumerate(test_dataset):
        if i % 100 == 0:
            print(f"On {i} of {len(test_dataset)} examples.")
        # Load through the model.
        file_name = test_dataset.files[i]
        img = img.to(args.device)
        mask = mask.to(args.device)

        output = model(torch.reshape(img, (1, 3, 224, 224)))

        # Convert output to labels

        if mask.sum().item() > 0:
            contains_mask = True
        else:
            contains_mask = False

        # Calculate IoU
        iou = calc_iou.update(output, mask)
        iou_results.append([file_name, contains_mask, iou.item()])
        # Calculate precision / recall
        prc.update(output, mask)

    # Create figure for the examples
    generate_examples(model, test_dataset, args.task, args.dump_path)
    # Save iou as pandas array
    iou_results.append(['total', None, calc_iou.value])
    df = pd.DataFrame(iou_results, columns=['file_name', 'contains_mask', 'iou'])
    df.to_csv(os.path.join(args.dump_path, 'iou.csv'))
    # Save precision recall as numpy array
    joblib.dump(prc, os.path.join(args.dump_path, 'prc.joblib'))
    create_precision_recall_curve(prc, args.dump_path)

@torch.no_grad()
def generate_examples(model, dataset, task, dump_path, threshold=0.5):

    # These examples were found to have high standard deviation in the calculated 
    # IoU values, so I will use them as the examples to look at
    print(task)
    if task == 'solar':
        examples = [5908, 5116, 5469, 6675, 6280, 6363, 4860, 4790, 6575, 4756]
        data_path = "/scratch/zach/solar-pv/"
        mask_path = "/scratch/zach/mask_tensors/"
    else:
        raise ValueError("This task is not supported for analysis")

    for example in examples:
        # Load through the model.
        img, mask = dataset[example]
        img1 = img.to(model.device)

        output = model(torch.reshape(img1, (1, 3, 224, 224)))

        # img = img.detach().cpu().reshape([1,3,224,224])
        output = output.detach().cpu().numpy().reshape(224, 224)

        file_name = dataset.files[example]
        img = Image.open(data_path + file_name)
        mask = torch.load(mask_path + file_name.split(".")[0] + '.pt')

        output[output < threshold] = 0
        output[output > threshold] = 1
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.xlabel("Image")
        plt.subplot(1, 3, 2)
        plt.imshow(mask[0])
        plt.xlabel("Actual mask")
        plt.subplot(1, 3, 3)
        plt.imshow(output)
        plt.xlabel("Predicted image")

        file_name = dataset.files[example].split(".")[0]

        plt.savefig(os.path.join(dump_path, file_name + ".png"))



def create_precision_recall_curve(prc, dump_path):
    precision = []
    recall = []
    for score in prc.scores:
        precision.append((score['tp'] / (score['tp'] + score['fp'])))
        recall.append((score['tp'] / (score['tp'] + score['fn'])))

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(dump_path, "precision_recall.png"))


class JaccardIndex():
    def __init__(self, threshold = 0.5):
        self.numerator = 0
        self.denominator = 0
        self.value = 0
        # In case we want to change the threshold
        self.threshold = threshold

    def update(self, preds, target):
        o_2 = copy.deepcopy(preds)
        o_2[o_2 < self.threshold] = 0
        o_2[o_2 > self.threshold] = 1

        intersection = (o_2 * target).sum()

        union = preds.sum() + target.sum() - intersection

        # Prevent dividing by zero
        if union == 0:
            return 0
        self.numerator += intersection
        self.denominator += union
        self.value = self.numerator / self.denominator

        return intersection / union

    def __repr__(self):
        return self.numerator / self.denominator


class PrecisionRecall():
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.scores = []
        # Create a place to store all of the scores
        for i in range(len(thresholds)):
            self.scores.append({
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0
            })

    def update(self, pred, target):
        all_ones = torch.ones(target.shape).to(target.device)
        for i, threshold in enumerate(self.thresholds):
            # Calculate the tp, fp, fn, tn for each of the
            # thresholds
            # Calculate mask for threshold
            output = copy.deepcopy(pred[0])
            output[output < threshold] = 0
            output[output > threshold] = 1
            # Calculate tp, fp, tn, fp and add it to
            # the scores.
            self.scores[i]['tp'] += (output * target).sum().type(torch.int).item()
            self.scores[i]['fp'] += (
                output * (all_ones - target)
            ).sum().type(torch.int).item()
            self.scores[i]['tn'] += (
                (all_ones - output) * (all_ones - target)
            ).sum().type(torch.int).item()
            self.scores[i]['fn'] += (
                (all_ones - output) * target
            ).sum().type(torch.int).item()


def create_train_val_curve(experiment_path, figsize=(5, 5)):
    df = pd.read_csv(os.path.join(experiment_path, "performance.csv"))

    train = df[df["stage"] == "train"]
    val = df[df["stage"] == "val"]

    plt.figure(figsize=figsize)
    plt.plot(train["epoch"], train["loss"], label="train")
    plt.plot(val["epoch"], val["loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss (BCE)")
    plt.legend()
    plt.savefig(os.path.join(experiment_path, "train_val.png"))
    plt.clf()


class Model:
    def __init__(self, data_path, device, model_type="best"):
        self.data_path = data_path
        self.model_type = model_type
        self.device = device
        
        self.encoder = encoders.load('none')
        self.encoder.load_state_dict(torch.load(
            os.path.join(self.data_path, "enc_" + model_type + ".pt")
        ))
        self.decoder = decoders.load('unet', self.encoder)

        self.decoder.load_state_dict(torch.load(
            os.path.join(self.data_path, "dec_" + model_type + ".pt")
        ))

        # Make sure the model is in evaluate mode.
        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def __call__(self, inp):

        output = self.encoder(inp)
        output = self.decoder(output)
        output = torch.sigmoid(output)

        return output

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script handles analysis creation."
    )
    parser.add_argument(
        "--task",
        choices=["solar"],
        type=str,
        help="The task of the experiment to analyze.",
        required=True
    )

    parser.add_argument("--experiment_path", type=str, help="The path from which to load model results.", required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="best",
        help="""By default, the best model is chosen. However,if 
        you save multiple models during training, you could select another model.""",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="imagenet",
        choices=["imagenet", "data"],
        required=True
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold between classes for IoU loss.",
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        help="Where to put analysis results.",
        required=True
    )

    args = parser.parse_args()

    main(args)
