#!/bin/bash
# This bash script iterates through the files and runs a gdal utility to retile
# the images into the desired sizes.
# For each of the files output, generate retiled images of size 224 x 224

large_images="/home/sl636/inria/AerialImageDataset/val_images/*.tif"

for image in $large_images; do
	echo Working on ${image}
	gdal_retile.py -targetDir /home/sl636/inria/AerialImageDataset/retiled_val_images/ -ps 224 224 $image
done

