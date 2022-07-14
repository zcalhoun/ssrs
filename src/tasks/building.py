import os
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset


class BuildingSegmentationDataset(Dataset):
    def __init__(self, path, images, mask_path, transform=None, augmentations=None):
        self.path = path
        self.images = images
        self.mask_path = mask_path
        self.transform = transform
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.files[idx]

        # TODO: ensure mask_name is right
        mask_name = self.mask_path+img_name+'.npy'

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

            # We still need to transform to a tensor and normalize.
            if self.transform:
                img = self.transform(img)
        else:
            img = Image.open(self.path + img_name)
            # Apply transforms
            if self.transform:
                img = self.transform(img)

        return img.type(torch.FloatTensor), mask.type(torch.LongTensor)
