#!/usr/bin/env bash

CONFIG="/data5/laiping/tianzhibei/code/Large-Selective-Kernel-Network/configs/lsknet/lsk_s_fpn_1x_dota_le90.py"
GPUS=2
NNODES=1
NODE_RANK=0
PORT=29500
MASTER_ADDR=127.0.0.1

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --seed 0 