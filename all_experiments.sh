#!/bin/bash


CUDA=0
TASK=building

for SIZE in 64 128 256 512 1024
do
    # # Run each experiment 3 times
    # for RUN in 1 2 3
    # do
    # python3 main.py --task $TASK \
    #     --epochs 100 \
    #     --device cuda:$CUDA \
    #     --dump_path /scratch/saad/$TASK/data-comparison/$SIZE/supervised/t$RUN \
    #     --encoder imagenet \
    #     --normalization imagenet \
    #     --fine_tune_encoder True \
    #     --batch_size 16 \
    #     --data_size $SIZE
    # done

    # # SWAV-Imagenet
    # for RUN in 1 2 3
    # do
    # python3 main.py --task $TASK \
    #     --epochs 100 \
    #     --device cuda:$CUDA \
    #     --dump_path /scratch/saad/$TASK/data-comparison/$SIZE/swav-imagenet/t$RUN \
    #     --encoder swav \
    #     --normalization imagenet \
    #     --fine_tune_encoder True \
    #     --batch_size 16 \
    #     --data_size $SIZE
    # done


    # SwAV-b1
    for RUN in 1 2 3
    do
    python3 main.py --task $TASK \
        --epochs 100 \
        --device cuda:$CUDA \
        --dump_path /scratch/saad/$TASK/data-comparison/$SIZE/swav-b1/t$RUN \
        --encoder swav-b1 \
        --normalization data \
        --fine_tune_encoder True \
        --batch_size 16 \
        --data_size $SIZE
    done
done