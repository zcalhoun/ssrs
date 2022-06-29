#!/bin/bash
# For each of the files output, generate retiled images of size 224 x 224

large_images="ls /scratch/zach/masks2/*.tif"

for image in $large_images; do
	echo Working on ${image}
	gdal_retile.py -targetDir /scratch/zach/masks_3/ -ps 224 224 -overlap 70 $image
done
