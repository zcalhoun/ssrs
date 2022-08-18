#!/bin/bash

CUDA=0
TASK=crop_delineation
for SIZE in 64 128 256 512 1024
do
    for RUN in 1 2 3
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path ./experiments/$TASK/data-comparison/$SIZE/swav-c2/t$RUN \
        --encoder swav-c2 \
        --normalization data \
        --fine_tune_encoder True \
        --batch_size 16 \
        --data_size $SIZE
    done
done
