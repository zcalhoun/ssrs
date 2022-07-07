import os 
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class SolarPVDataset(Dataset):
    def __init__(self, path, files, mask_path, transform=None, augmentations=None):
        self.path = path
        self.files = files
        self.mask_path = mask_path
        self.transform = transform
        self.aug = augmentations
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.files[idx]

        mask_name = self.mask_path+img_name.split(".")[0]+'.pt'

        if os.path.exists(mask_name):
            mask = torch.load(mask_name)
        else:
            mask = torch.zeros((1,224,224))

        # Apply albumentations augmentations
        # before applying the standard transform
        if self.aug:
            # Must convert to numpy array for
            # this to work nicely with albumentations
            img = cv2.imread(self.path+img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
            mask = mask.numpy().T
            augmented = self.aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].copy().T

            augmented = self.to_tensor(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        else:
            img = Image.open(self.path+img_name)
            # Apply transforms
            if self.transform:
                img = self.transform(img)


        return img.type(torch.FloatTensor), mask.type(torch.LongTensor)

