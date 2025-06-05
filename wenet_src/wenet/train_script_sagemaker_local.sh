#!/bin/bash

MODEL="/tmp/pretrain_model"
SM_NUM_GPUS=1
NODE_NUMBER=1
NODE_INDEX=0
# SM_MASTER_ADDR=
DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"
DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS"
torchrun ${DISTRIBUTED_ARGS} bin/train.py --config ../FireRedASR-AED-L_wenet/train.yaml --model_dir ../../trianing_result --train_data /wenet_finetuning/data/zh/train.list --cv_data /wenet_finetuning/data/zh/test.list --ddp.dist_backend 'nccl' --prefetch 16 --num_workers 16 --checkpoint ../FireRedASR-AED-L_wenet/wenet_firered.pt --use_amp


if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi
