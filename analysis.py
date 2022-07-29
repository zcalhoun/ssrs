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

from sklearn.metrics import precision_recall_curve

from PIL import Image

from src import datasets
from models import encoders, decoders

import torch


def main(args):

    # Check if directory exists. If not, then create the directory

    if ~os.path.exists(args.dump_path):
        print("Path doesn't exist")
        os.makedirs(args.dump_path)

    # First off, create the train/val curve.
    create_train_val_curve(args.experiment_path)

    # You only need the test dataset output.
    _, test_dataset = datasets.load(task=args.task, normalization=args.normalization, old=False, size=args.data_size)

    model = Model(args.experiment_path, args.device, args.model_type)
    model.to(args.device)

    preds = []
    targets = []
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
        with torch.no_grad():
            output = model(torch.reshape(img, (1, 3, 224, 224)))

        # Convert output to labels
        if mask.sum().item() > 0:
            contains_mask = True
        else:
            contains_mask = False
            if output.sum().item() == 0:
                iou_results.append([file_name, contains_mask, 0])
                continue
            
        # Calculate IoU
        iou = calc_iou.update(output, mask)
        # pdb.set_trace()
        iou_results.append([file_name, contains_mask, iou])
        # Calculate precision / recall
        preds.append(output.cpu().numpy().flatten())
        targets.append(mask.cpu().numpy().flatten())
        # prc.update(output, mask)

    print(f"Experiment path: {args.experiment_path}")
    print(f"IoU: {calc_iou.value}")

    # generate_examples(model, test_dataset, args.task, args.dump_path)
    # Save iou as pandas array
    iou_results.append(['total', None, calc_iou.value])
    df = pd.DataFrame(iou_results, columns=['file_name', 'contains_mask', 'iou'])
    df.to_csv(os.path.join(args.dump_path, 'iou.csv'))

    create_precision_recall_curve(preds, targets, args.dump_path)


def create_images_only(args):
    """
    This function is meant as a helpful way to generate images when you
    find more interesting examples you want to look at.
    """
    test_dataset = datasets.load(task=args.task, evaluate=True, normalization=args.normalization, old=False)
    model = Model(args.experiment_path, args.device, args.model_type)
    model.to(args.device)
    generate_examples(model, test_dataset, args.task, args.dump_path)


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


def create_precision_recall_curve(preds, targets, dump_path):
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    precision, recall, t = precision_recall_curve(targets, preds)
    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(dump_path, "precision_recall.png"))

    df = pd.DataFrame({
        'recall': recall,
        'precision': precision
    })

    max_f1 = ((2 * precision * recall) / (precision + recall)).max()

    print(f"The max F1 score is {max_f1}.")
    df.to_csv(os.path.join(dump_path, "precision_recall.csv"))


class JaccardIndex():
    def __init__(self, threshold=0.5):
        self.numerator = 0
        self.denominator = 0
        self.value = 0
        # In case we want to change the threshold
        self.threshold = threshold

    def update(self, preds, target):
        o_2 = copy.deepcopy(preds)
        o_2[o_2 < self.threshold] = 0
        o_2[o_2 >= self.threshold] = 1

        intersection = (o_2 * target).sum()

        union = o_2.sum() + target.sum() - intersection

        # Prevent dividing by zero
        if union == 0:
            return 0
        self.numerator += intersection
        self.denominator += union
        self.value = self.numerator / self.denominator

        return (intersection / union).item()

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
        if (pred.sum() < self.thresholds.min()) and (target.sum() == 0):
            # Don't worry about calculating true negatives.
            return

        for i, threshold in enumerate(self.thresholds):
            # Calculate the tp, fp, fn, tn for each of the
            # thresholds
            # Calculate mask for threshold
            output = copy.deepcopy(pred[0])
            output[output < threshold] = 0
            output[output >= threshold] = 1
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

    def get_max_f1(self,):
        best_f1 = 0
        eps = 1e-6 # Needed to prevent dividing by zero
        for score in self.scores:
            precision = (score['tp'] + eps) / (score['tp'] + score['fp'] + eps)
            recall = score['tp'] / (score['tp'] + score['fn'])
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1


def create_train_val_curve(experiment_path, figsize=(5, 5)):
    df = pd.read_csv(os.path.join(experiment_path, "performance.csv"))

    train = df[df["stage"] == "train"]
    val = df[df["stage"] == "val"]

    plt.figure(figsize=figsize)
    plt.plot(train["epoch"], train["loss"], label="train")
    plt.plot(val["epoch"], val["loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss (1 - SoftIoU)")
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
        choices=["solar", "crop_delineation"],
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

    parser.add_argument(
        "--generate_examples_only",
        type=bool,
        default=False,
        help="If you only want to generate samples rather than calculate statistics."
    )

    parser.add_argument(
        "--data_size",
        type=str,
        default='normal',
        help="Use this if you want to look at a smaller dataset."
    )

    args = parser.parse_args()

    if args.generate_examples_only:
        create_images_only(args)
    else:
        main(args)
