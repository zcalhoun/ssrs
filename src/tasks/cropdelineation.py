"""Field/crop delineation"""
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset


class CropDelineationDataset(Dataset):
    def __init__(self, path, files, mask_filled, transform=None, augmentations=None):
        self.img_path = path + 'imgs/'  # Path to images
        self.files = files  # files in the image path
        self.mask_filled = mask_filled  # Path masks
        if mask_filled:
            self.mask_path = path + 'masks/'
        else:
            self.mask_path = path + 'masks_filled/'
        self.transform = transform
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image and mask
        img_name = self.img_path + str(self.files[idx]) + '.jpeg'
        mask_name = self.mask_path + str(self.files[idx]) + '.png'

        # Apply albumentations augmentations
        # before applying the standard transform
        if self.aug:
            # Must convert to numpy array for
            # this to work nicely with albumentations
            # cv2 is the recommended method for reading images.
            # although you could just convert a PIL image to
            # a numpy array, too.
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Albumentations works more nicely if you
            # have the mask as a 2D array
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255

            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # Albumentations works more nicely if you
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        else:

            img = Image.open(img_name).convert('RGB')
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)

            mask = np.array(Image.open(mask_name)) / 255

        return img.type(torch.FloatTensor), torch.from_numpy(mask).type(torch.LongTensor)
