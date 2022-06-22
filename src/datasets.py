import os
import logging

from tasks.solar import SolarPVDataset


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
    data_path = "/data/users/zdc6/data/SolarPV/fresno"
    files = list(filter(lambda x: x[-3:] == "tif", os.listdir(data_path)))
    logging.debug(f"There are {len(files)} files in the Frenso dataset.")
    shape_file = "/data/users/zdc6/data/SolarPV/SolarArrayPolygons.geojson"

    # Split files into two lists with an 80/20 split.
    # In this case, put every fifth file into the test set.
    mask = np.arange(0, len(files)) % 5 == 0
    test_files = files[mask]
    train_files = files[~mask]

    # Load the training dataset
    logging.debug("Creating the training dataset.")
    train_dataset = SolarPVDataset(data_path, train_files, shape_file)
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    test_dataset = SolarPVDataset(data_path, test_files, shape_file)
    # Return the training and test dataset
    return train_dataset, test_dataset
