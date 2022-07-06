#!/bin/bash

python3 main.py --task solar \
	--epochs 100 \
	--device cuda:1 \
	--dump_path ./experiments/solar/t4 \
	--fine_tune_encoder True \
	--batch_size 16
