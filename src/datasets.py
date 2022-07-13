import os
import logging
import numpy as np
import joblib

from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .tasks.solar import SolarPVDataset


def load(task, normalization="data", augmentations=False, evaluate=False, old=False):
    logging.debug(f"In datasets, the task {task} is being loaded.")
    task_map = {
        "solar": _load_solar_data(normalization, augmentations, evaluate, old),
    }
    if task not in task_map:
        logging.error(f"{task} not implemented.")
        raise NotImplementedError("This task is not supported at this time.")
    return task_map[task]


def _load_solar_data(normalization, augmentations, evaluate, old=False):
    # Split the data into a train and test set
    data_path = "/scratch/zach/solar-pv/"
    mask_path = "/scratch/zach/mask_tensors/"
    files = joblib.load("/scratch/zach/train_test_split.joblib")
    logging.debug(f"There are {len(files)} files in the Frenso dataset.")

    # Split files into two lists with an 80/20 split.
    # In this case, put every fifth file into the test set.
    if evaluate:
        test_files = files["test"]["mask"]
        train_files = files["train"]["mask"]
    else:
        test_files = files["test"]["empty"] + files["test"]["mask"]
        train_files = files["train"]["empty"] + files["train"]["mask"]

    # We want to ensure that the normalization scheme is considered. In the case
    # that we are using a pretrained method, it might be better to use that
    # pretrained normalization mean and standard deviation. Otherwise, it would
    # be better to use the normalization scheme applied on the original dataset.
    # By specifying 'data', we are saying that we want to use the calculated
    # mean and standard deviation on the dataset. Other normalization methods
    # should be added as more pre-trained methods are supported.
    if normalization == "data":
        # This normalization was calculated by taking several sample
        # images (as tensors) and calculating the average RGB value along with the
        # standard deviation.
        print("Normalizing using the data.")
        if old:
            print("Using old normalization")
            normalize = {'mean': [0.494, 0.491, 0.499], 'std': [0.142, 0.141, 0.135]}
        else:
            print("Using new normalization")
            normalize = {"mean": [0.507, 0.513, 0.461], "std": [0.172, 0.133, 0.114]}
    elif normalization == "imagenet":
        print("Normalize using imagenet.")
        # This normalization scheme uses the means and weights for ImageNet.
        normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    else:
        raise NotImplementedError("This normalization scheme isn't supported.")

    # Include everything but the kitchen sink...
    # we want a large collection of transformations.
    # The first four ensure that each image is represented
    # in 8 separate ways...the next 3 transformations
    # affect the coloration and pixel values, whereas
    # the final 2 ensure that the image is in a format
    # that the model likes.
    if augmentations:
        print("Adding augmentations...")
        aug = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Transpose(),
                A.RandomRotate90(),
            ]
        )

    tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])

    train_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    # Load the training dataset
    logging.debug("Creating the training dataset.")
    if augmentations:
        train_dataset = SolarPVDataset(
            data_path,
            train_files,
            mask_path,
            transform=train_transform,
            augmentations=aug,
        )
    else:
        train_dataset = SolarPVDataset(
            data_path, train_files, mask_path, transform=train_transform
        )
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    test_dataset = SolarPVDataset(
        data_path, test_files, mask_path, transform=test_transform
    )
    # Return the training and test dataset
    return train_dataset, test_dataset
