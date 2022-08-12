#!/bin/bash

SIZE=512
CUDA=0

# Run each experiment 3 times
for RUN in 1 2 3
do
python3 main.py --task solar \
	--epochs 100 \
	--device cuda:$CUDA \
	--dump_path ./experiments/solar/data-comparison/$SIZE/supervised/t$RUN \
	--encoder imagenet \
	--normalization imagenet \
    --fine_tune_encoder True \
	--batch_size 16 \
    --data_size $SIZE
done

# SWAV-Imagenet
for RUN in 1 2 3
do
python3 main.py --task solar \
	--epochs 100 \
	--device cuda:$CUDA \
	--dump_path ./experiments/solar/data-comparison/$SIZE/swav-imagenet/t$RUN \
	--encoder swav \
	--normalization imagenet \
    --fine_tune_encoder True \
	--batch_size 16 \
    --data_size $SIZE
done


# SWAV-S3
for RUN in 1 2 3
do
python3 main.py --task solar \
	--epochs 100 \
	--device cuda:$CUDA \
	--dump_path ./experiments/solar/data-comparison/$SIZE/swav-s3/t$RUN \
	--encoder swav-s3 \
	--normalization data \
    --fine_tune_encoder True \
	--batch_size 16 \
    --data_size $SIZE
done