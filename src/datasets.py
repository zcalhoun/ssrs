import os
import logging
import numpy as np
import joblib

from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .tasks.solar import SolarPVDataset


def load(task, augmentations=False):
    logging.debug(f"In datasets, the task {task} is being loaded.")
    task_map = {
        "solar": _load_solar_data(augmentations),
    }
    if task not in task_map:
        raise ValueError
    return task_map[task]


def _load_solar_data(augmentations):
    # Split the data into a train and test set
    data_path = "/scratch/zach/solar-pv/"
    mask_path = "/scratch/zach/mask_tensors/"
    files = joblib.load("/scratch/zach/train_test_split.joblib")
    # files = np.array(list(filter(lambda x: x[-3:] == "tif", os.listdir(data_path))))
    logging.debug(f"There are {len(files)} files in the Frenso dataset.")

    # Split files into two lists with an 80/20 split.
    # In this case, put every fifth file into the test set.
    # mask = np.arange(0, len(files)) % 5 == 0
    test_files = files['test']['empty'] + files['test']['mask']
    train_files = files['train']['empty'] + files['train']['mask']

    # TODO - find the mean and standard deviation
    tr_normalize = transforms.Normalize(
        mean=[0.494, 0.491, 0.499], std=[0.142, 0.141, 0.135]
    )

    if augmentations:
        print("Adding augmentations...")
        aug = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.RandomRotate90(),
            A.PixelDropout(),
            A.GaussianBlur(),
            A.ColorJitter(),
            A.Normalize(
                mean=[0.494, 0.491, 0.499], std=[0.142, 0.141, 0.135]
            ),
			ToTensorV2(),
        ])

    # Resize shouldn't typically be necessary...but just in case,
    # the resize operation is included.
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            tr_normalize
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), tr_normalize]
    )

    # Load the training dataset
    logging.debug("Creating the training dataset.")
    if augmentations:
        train_dataset = SolarPVDataset(
            data_path, train_files, mask_path, transform=train_transform, augmentations=aug
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
