import os 
from PIL import Image

import torch
from torch.utils.data import Dataset


class SolarPVDataset(Dataset):
    def __init__(self, path, files, mask_path, transform=None):
        self.path = path
        self.files = files
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.files[idx]

        # Open image with PIL
        img = Image.open(self.path+img_name)
        mask_name = self.mask_path+img_name.split(".")[0]+'.pt'

        if os.path.exists(mask_name):
            mask = torch.load(mask_name)
        else:
            mask = torch.zeros((1,224,224))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, mask

