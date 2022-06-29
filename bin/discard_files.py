"""
This file deletes images that do not meet the desired image
sizes for the script to run properly.
"""

import os
from PIL import Image

data_path = '/scratch/zach/masks/'

def main():
	for i, file_path in enumerate(file_iter()):
		print(f"Deleting file {i}")
		os.remove(file_path)

def file_iter():
	files = os.listdir(data_path)
	for i, fn in enumerate(files):
		if i % 10000 == 0:
			print(f"On file {i}")
		with Image.open(data_path+fn) as f:
			if f.width !=224 or f.height !=224:
				yield data_path+fn


if __name__ == "__main__":
	main()