import rasterio
from rasterio.mask import mask
import geopandas as gpd

import torch
from torch.utils.data import Dataset


class SolarPVDataset(Dataset):
    def __init__(self, path, files, shape_file, transform=None):
        self.path = path
        self.files = files

        # Set up shape file
        # shape_path = '../../data/SolarPV/SolarArrayPolygons.geojson'
        self.df = gpd.read_file(shape_file).to_crs(epsg=26911)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.files[idx]

        # Load the data
        with rasterio.open(self.path + img_name) as data:
            # Load the mask
            out_mask, out_transform = mask(data, self.df["geometry"])
            img = data.read()

        # Remove the last layer for images that have a fourth band.
        img = img[:3, :, :]
        # Apply transforms
        if self.transform:
            img = self.transform(img.T)

        # Turn the mask into a binary mask
        out_mask = out_mask.sum(axis=0)
        out_mask[out_mask > 0] = 1

        # Turn both arrays into tensors
        out_mask = torch.from_numpy(out_mask.astype("float32")).reshape(1, 224, 224)
        return img, out_mask

    def show_item(self, idx):
        # Load the image
        img_name = self.files[idx]

        # Load the data
        with rasterio.open(self.path + img_name) as data:
            # Load the mask
            out_img, out_transform = mask(data, self.df["geometry"])

            return data.read(), out_img
