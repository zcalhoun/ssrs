import os
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


class SolarPVDataset(Dataset):
    def __init__(self, path, files, mask_path, transform=None, augmentations=None):
        self.path = path # Path to images
        self.files = files # files in the image path
        self.mask_path = mask_path # Path masks
        self.transform = transform
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.files[idx]

        mask_name = self.mask_path + img_name.split(".")[0] + ".pt"

        if os.path.exists(mask_name):
            mask = torch.load(mask_name)
        else:
            mask = torch.zeros((1, 224, 224))

        # Apply albumentations augmentations
        # before applying the standard transform
        if self.aug:
            # Must convert to numpy array for
            # this to work nicely with albumentations
            # cv2 is the recommended method for reading images.
            # although you could just convert a PIL image to
            # a numpy array, too.
            img = cv2.imread(self.path + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Albumentations works more nicely if you
            # have the mask as a 2D array
            mask = mask.numpy()[0]
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        else:
            # This piece of code is left so that the test set
            # gets a simple transform -- and as a legacy piece of
            # code in case you don't want to transform on the training
            # set with more than the required transforms...the code
            # could be refactored to only use albumentations, but
            # that will be left as a future exercise, since performance
            # shouldn't be impacted by leaving this code in.
            #
            # Now that augmentations are set up, we pretty much
            # should never have the augmentations excluded.
            # However, if you want to run an experiment without
            # the standard augmentations, and just with the required
            # augmentations needed to run the model, then turn
            # augmentations off.
            img = Image.open(self.path + img_name)
            # Apply transforms
            if self.transform:
                img = self.transform(img)

        # Type change required.
        return img.type(torch.FloatTensor), mask.type(torch.LongTensor)
