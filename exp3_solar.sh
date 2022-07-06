#!/bin/bash

python3 main.py --task solar \
	--epochs 100 \
	--device cuda:0 \
	--dump_path ./experiments/solar/t3 \
	--batch_size 16
