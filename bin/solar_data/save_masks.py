"""
DO NOT USE.

This script shows how to generate images using rasterio, but it takes forever
to run, and the resulting masks appear to be off. 

However, I'll keep this script for posterity, as it does demonstrate how you could
generate the masks, if you needed to.
"""

import os
import numpy as np

import rasterio
from rasterio.mask import mask
import geopandas as gpd

def main():
    
    data_path = "/scratch/zach/solar-pv/"
    shape_file = "/data/users/zdc6/data/SolarPV/SolarArrayPolygons.geojson"
    files = np.array(list(filter(lambda x: x[-3:] == "tif", os.listdir(data_path))))

    target_dir = "/scratch/zach/masks4/"

    df = gpd.read_file(shape_file).to_crs(epsg=26911)

    for i, fn in enumerate(files):
        if i % 10 == 0:
            print(f"Processed image {i} of {len(files)}")

        with rasterio.open(data_path+fn) as src:
            out_image, out_transform = mask(src, df['geometry'])
            out_meta = src.meta.copy()

        out_meta.update({
            "height": out_image.shape[1],
            "width" : out_image.shape[2]
        })

        with rasterio.open(target_dir+fn, 'w', **out_meta) as dest:
            dest.write(out_image)
        
        # # To avoid saving a bunch of files that 
        # # are empty, only save the mask if it
        # # contains a positive sample.
        # out_mask = out_mask.sum(axis=0)
        # out_mask[out_mask > 0] = 1

        # if out_mask.sum() > 0:
        #     out_mask = out_mask.astype(int)
        #     mask_name = fn.split(".")[0]
        #     print(f"Saving mask {mask_name}")
        #     np.save(target_dir+mask_name+'.npy', out_mask)


if __name__ == "__main__":
    main()
