"""
This script didn't end up being used (I ran the code in a Jupyter notebook)
but this code is a good example of how to generate masks for a set of images
given a CSV version of the images.

Basically, the code iterates through the images, loads the mask, and creates the
mask.

Further processing may be required to ensure that the images are formatted correctly.
"""

import os
import argparse

from PIL import Image, ImageDraw

import pandas as pd

parser = argparse.ArgumentParser(description="Generate masks for each image.")

parser.add_argument("--dump_path", type=str, help="Where to save the files.")


def main(dump_path):

    # Get list of masks that we want to generate:
    fresno_dir = "/home/zdc6/data/SolarPV/fresno/"
    files = list(filter(lambda x: x.split(".")[-1] == "tif", os.listdir(fresno_dir)))
    print(f"{len(files)} files to read...")
    # Get references
    image_map = pd.read_csv(
        "/home/zdc6/data/SolarPV/polygonDataExceptVertices.csv", index_col=0
    ).set_index("image_name")

    polygon_vertices = pd.read_csv(
        "/home/zdc6/data/SolarPV/polygonVertices_PixelCoordinates.csv"
    ).set_index("polygon_id")

    # Iterate through each image.
    for fn in files:
        print(f"Reading file {fn}...")
        image_name = fn.split(".")[0]
        # Get polygons for this image.
        try:
            polygons = image_map.loc[image_name]
        except KeyError:
            print(f"No polygons found for image {image_name}")
            continue

        # Create new blank image and drawing
        new_img = Image.new(str(1), (5000, 5000), color=0)
        mask = ImageDraw.Draw(new_img)

        # Loop through all of the polygons:
        try:
            for idx, polygon in polygons.iterrows():
                vertices = polygon_vertices.loc[int(polygon["polygon_id"])]
                vertices = [i for i in vertices[1:] if not (pd.isna(i))]
                mask.polygon(vertices, fill=1)
        except AttributeError:
            # If there is no attribute, don't iterate through
            vertices = polygon_vertices.loc[int(polygons["polygon_id"])]
            vertices = [i for i in vertices[1:] if not (pd.isna(i))]
            mask.polygon(vertices, fill=1)

        print("Saving image...")
        new_img.save(dump_path + fn)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dump_path)
