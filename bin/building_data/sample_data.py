import numpy as np
import os
import shutil
from natsort import natsorted

mask_path = "/home/sl636/inria/AerialImageDataset/retiled_train_masks/"
img_path = "/home/sl636/inria/AerialImageDataset/retiled_train_images/"

mask_val_path = "/home/sl636/inria/AerialImageDataset/retiled_val_masks/"
img_val_path = "/home/sl636/inria/AerialImageDataset/retiled_val_images/"


images = natsorted(os.listdir(img_path))
masks = natsorted(os.listdir(mask_path))
val_images = natsorted(os.listdir(img_val_path))
val_masks = natsorted(os.listdir(mask_val_path))


def pick_n_examples(n):

    rng = np.random.default_rng(123)
    rints = rng.integers(low=0, high=len(images), size=n)

    train_path = f"/scratch/saad/{n}_images/"
    train_mask_path = f"/scratch/saad/{n}_masks/"

    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(train_mask_path):
        os.mkdir(train_mask_path)

    for r in rints:
        shutil.copy(img_path+images[r], train_path)
        shutil.copy(mask_path+masks[r], train_mask_path)

    rng = np.random.default_rng(123)
    rints_val = rng.integers(low=0, high=len(val_images), size=n)

    val_path = f"/scratch/saad/{n}_val_images/"
    val_mask_path = f"/scratch/saad/{n}_val_masks/"

    if not os.path.isdir(val_path):
        os.mkdir(val_path)

    if not os.path.isdir(val_mask_path):
        os.mkdir(val_mask_path)

    for r in rints_val:
        shutil.copy(img_val_path+val_images[r], val_path)
        shutil.copy(mask_val_path+val_masks[r], val_mask_path)

    print(len(set(os.listdir(train_path)).difference(
        set(os.listdir(train_mask_path)))))
    print(len(set(os.listdir(val_path)).difference(set(os.listdir(val_mask_path)))))


for n in [64, 128, 256, 512, 1024]:
    pick_n_examples(n)
