import os
import logging
import numpy as np
import pandas as pd
import joblib

from torchvision import transforms
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .tasks.solar import SolarPVDataset
from .tasks.building import BuildingSegmentationDataset
from .tasks.cropdelineation import CropDelineationDataset


def load(task, normalization="data", augmentations=False, data_size=1024, evaluate=False, old=False, size="small"):
    logging.debug(f"In datasets, the task {task} is being loaded.")
    if task == "solar":
        print("Loading solar dataset.")
        return _load_solar_data(normalization, augmentations, evaluate, old, data_size)
    elif task == "building":
        print("Loading building dataset.")
        return _load_building_data(normalization, augmentations, data_size)
    elif task == "crop_delineation":
        print("Loading crop delineation dataset.")
        return _load_cropdel_data(normalization, augmentations, evaluate, data_size)

def _load_cropdel_data(normalization, augmentations, evaluate, size=None):
    print(f"Data evaluate: {evaluate}")
    """
    This function takes care of loading the crop segmentation
    data for training the model.
    """
    # Change this to false if you want to use a different set of masks
    mask_filled = False
    data_path = "/scratch/crop-delineation/data/"
    # This loads the list of files to reference
    file_map = pd.read_csv(data_path + "clean_data.csv")
    if size is not None:
        print(f"Loading crop delineation training data with size {size}.")
        train_files = list(joblib.load(data_path + f"train_{size}.joblib"))
    else:
        print("Loading the complete training dataset.")
        train_files = list(file_map[file_map['split'] == 'train']['indices'])
    
    val_files = list(file_map[file_map['split'] == 'val']['indices'])
    test_files = list(file_map[file_map['split'] == 'test']['indices'])
    if normalization == "data":
        # TODO -- calculate this
        normalize = {"mean": [0.238, 0.297, 0.317], "std": [0.187, 0.123, 0.114]}
    elif normalization == 'all':
        print("Normalize using all remote sensing data.")
        normalize = {
            'mean': [0.431, 0.449, 0.411],
            'std': [0.120, 0.175, 0.164]
        }
    elif normalization == 'imagenet':
        normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    
    # Add augmentations
    if augmentations:
        print("Adding augmentations...")
        aug = A.Compose(
            [
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Transpose(),
                A.Normalize(
                    mean=normalize['mean'],
                    std=normalize['std']
                ),
                ToTensorV2()
            ]
        )

    tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])

    train_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    # Create the train dataset
    logging.debug("Creating the training dataset.")
    if augmentations:
        train_dataset = CropDelineationDataset(
            data_path,
            train_files,
            mask_filled,
            transform=train_transform,
            augmentations=aug,
        )
    else:
        train_dataset = CropDelineationDataset(
            data_path, train_files, mask_filled, transform=train_transform
        )
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    val_dataset = CropDelineationDataset(
        data_path, val_files, mask_filled, transform=test_transform
    )
    # Return the training and test dataset
    test_dataset = CropDelineationDataset(
        data_path, test_files, mask_filled, transform=test_transform
    )
    if evaluate:
        return test_dataset
    else:
        return train_dataset, val_dataset


def _load_solar_data(normalization, augmentations, evaluate, old=False, data_size="normal"):
    # Split the data into a train and test set
    data_path = "/scratch/zach/solar-pv/"
    mask_path = "/scratch/zach/mask_tensors/"

    if data_size != 'normal':
        val_files = joblib.load("/scratch/zach/train_test_split_1024.joblib")
        print(f"Size: {data_size}")
        if data_size == "64":
            print("Loading dataset with 64 training examples")
            files = joblib.load("/scratch/zach/train_test_split_64.joblib")
        elif data_size == "128":
            print("Loading dataset with 128 training examples")
            files = joblib.load("/scratch/zach/train_test_split_128.joblib")
        elif data_size == "256":
            print("Loading dataset with 256 training examples")
            files = joblib.load("/scratch/zach/train_test_split_256.joblib")
        elif data_size == "512":
            print("Loading dataset with 512 training examples")
            files = joblib.load("/scratch/zach/train_test_split_512.joblib")
        elif data_size == "1024":
            print("Loading dataset with 1024 training examples")
            files = joblib.load("/scratch/zach/train_test_split_1024.joblib")

        # Make sure that the test set is the same for all trials.
        files['test']['mask'] = val_files['test']['mask']
        files['test']['empty'] = val_files['test']['empty']
    else:
        print("Loading full dataset")
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
    elif normalization == 'all':
        print("Normalize using all remote sensing data.")
        normalize = {
            'mean': [0.431, 0.449, 0.411],
            'std': [0.120, 0.175, 0.164]
        }
    elif normalization == "building":
        print("Normalize using the building statistics.")
        normalize = {
            'mean': [0.406, 0.428, 0.394],
            'std': [0.201, 0.183, 0.176]
        }
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
                # A.VerticalFlip(),
                # A.Transpose(),
                # A.RandomRotate90(),
                A.Normalize(
                    mean=normalize['mean'],
                    std=normalize['std']
                ),
                ToTensorV2()
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


def _load_building_data(normalization, augmentations, data_size):
    # Paths to train and test set (as split from INRIA)
    # train_imgs_path = "/scratch/saad/1000_images/"
    # train_masks_path = "/scratch/saad/1000_masks/"
    # val_imgs_path = "/scratch/saad/1000_val_images/"
    # val_masks_path = "/scratch/saad/1000_val_masks/"

    train_imgs_path = f"/scratch/saad/{data_size}_images/"
    train_masks_path = f"/scratch/saad/{data_size}_masks/"
    val_imgs_path = f"/scratch/saad/{data_size}_val_images/"
    val_masks_path = f"/scratch/saad/{data_size}_val_masks/"

    train_imgs = natsorted(os.listdir(train_imgs_path))
    val_imgs = natsorted(os.listdir(val_imgs_path))

    logging.debug(
        f"We are using {len(train_imgs)} training images and {len(val_imgs)} validation images from the INRIA building dataset.")

    # We want to ensure that the normalization scheme is considered. In the case
    # that we are using a pretrained method, it might be better to use that
    # pretrained normalization mean and standard deviation. Otherwise, it would
    # be better to use the normalization scheme applied on the original dataset.
    # By specifying 'data', we are saying that we want to use the calculated
    # mean and standard deviation on the dataset. Other normalization methods
    # should be added as more pre-trained methods are supported.
    if normalization == 'data':
        # This normalization was calculated by taking several sample
        # images (as tensors) and calculating the average RGB value along with the
        # standard deviation.
        print("Normalizing using the data.")
        normalize = {
            'mean': [0.406, 0.428, 0.394],
            'std': [0.201, 0.183, 0.176]
        }
    elif normalization == 'imagenet':
        print("Normalize using imagenet.")
        # This normalization scheme uses the means and weights for ImageNet.
        normalize = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    elif normalization == 'all':
        print("Normalize using all remote sensing data.")
        normalize = {
            'mean': [0.431, 0.449, 0.411],
            'std': [0.120, 0.175, 0.164]
        }
    elif normalization == 'solar':
        print("Normalize using the solar data statistics.")

        normalize = {"mean": [0.507, 0.513, 0.461], "std": [0.172, 0.133, 0.114]}
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
        aug = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.RandomRotate90(),
        ])

    tr_normalize = transforms.Normalize(
        mean=normalize['mean'], std=normalize['std']
    )

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
        train_dataset = BuildingSegmentationDataset(
            train_imgs_path, train_imgs, train_masks_path,
            transform=train_transform, augmentations=aug
        )
    else:
        train_dataset = BuildingSegmentationDataset(
            train_imgs_path, train_imgs, train_masks_path,
            transform=train_transform
        )
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    test_dataset = BuildingSegmentationDataset(
        val_imgs_path, val_imgs, val_masks_path,
        transform=test_transform
    )
    # Return the training and test dataset
    return train_dataset, test_dataset