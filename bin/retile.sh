#!/bin/bash
# This bash script iterates through the files and runs a gdal utility to retile
# the images into the desired sizes.
# For each of the files output, generate retiled images of size 224 x 224

large_images="ls /scratch/zach/masks_raw/*.tif"

for image in $large_images; do
	echo Working on ${image}
	gdal_retile.py -targetDir /scratch/zach/masks/ -ps 224 224 -overlap 70 $image
done
