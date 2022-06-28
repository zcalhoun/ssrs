import os
import logging
import numpy as np

from torchvision import transforms

from .tasks.solar import SolarPVDataset


def load(task):
    logging.debug(f"In datasets, the task {task} is being loaded.")
    task_map = {
        "solar": _load_solar_data,
    }
    if task not in task_map:
        raise ValueError
    return task_map[task]()


def _load_solar_data():
    # Split the data into a train and test set
    data_path = "/scratch/zach/solar-pv/"
    files = np.array(list(filter(lambda x: x[-3:] == "tif", os.listdir(data_path))))
    logging.debug(f"There are {len(files)} files in the Frenso dataset.")
    shape_file = "/data/users/zdc6/data/SolarPV/SolarArrayPolygons.geojson"

    # Split files into two lists with an 80/20 split.
    # In this case, put every fifth file into the test set.
    mask = np.arange(0, len(files)) % 5 == 0
    test_files = files[mask]
    train_files = files[~mask]

    # TODO - find the mean and standard deviation
    tr_normalize = transforms.Normalize(
        mean=[0.494, 0.491, 0.499], std=[0.142, 0.141, 0.135]
    )

    # Resize shouldn't typically be necessary...but just in case,
    # the resize operation is included.
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            tr_normalize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), tr_normalize]
    )

    # Load the training dataset
    logging.debug("Creating the training dataset.")
    train_dataset = SolarPVDataset(
        data_path, train_files, shape_file, transform=train_transform
    )
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    test_dataset = SolarPVDataset(
        data_path, test_files, shape_file, transform=test_transform
    )
    # Return the training and test dataset
    return train_dataset, test_dataset
