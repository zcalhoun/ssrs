#!/bin/bash

CUDA=1
# Run each experiment 3 times
for SIZE in 64 128 256 512 1024
do
    for RUN in 1 2 3
    do
    python3 main.py --task solar \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path ./experiments/solar/data-comparison/$SIZE/swav-s4/t$RUN \
        --encoder swav-s4 \
        --normalization data \
        --fine_tune_encoder True \
        --batch_size 16 \
        --data_size $SIZE
    done
done
