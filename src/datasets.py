import os

from tasks.solar import SolarPVDataset


def load(task, test_size):
    task_map = {
        "solar": _load_solar_data,
    }
    if task not in task_map:
        raise ValueError
    return task_map[task]()


# TODO -- finish loading the solar data by splitting
# into a train/test split.
def _load_solar_data(test_size=0.2):
    # Split the data into a train and test set
    list(filter(lambda x: x[-3:] == "tif", os.listdir(path)))

    # Load the training dataset

    # Load the test dataset

    # Return the training and test dataset

