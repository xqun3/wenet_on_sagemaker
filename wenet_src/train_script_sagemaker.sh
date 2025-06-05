#!/bin/bash

MODEL="/tmp/pretrain_model"

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

torchrun ${DISTRIBUTED_ARGS} wenet/bin/train.py --config /tmp/model/train_modefied.yaml --model_dir /opt/ml/checkpoints --train_data /opt/ml/input/data/zh/train.list --cv_data /opt/ml/input/data/zh/test.list --ddp.dist_backend 'nccl' --prefetch 16 --num_workers 16 --checkpoint /tmp/model/wenet_firered.pt --use_amp


if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi
