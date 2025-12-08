#!/bin/bash

# Training configuration
EXP_ID=0
GPU_NUM=0
DDP=1
NUM_EPOCH=100
BATCH_SIZE=12
LEARNING_RATE=0.0001
LR_SCHEDULING=0
LR_SCHEDULER_TYPE='OnecycleLR'
NUM_CORES=8

# DDP configuration
NPROC_PER_NODE=6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
HOST_NODE_ADDR=10000

# Training
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    --max_restarts=3 \
    --rdzv_id=$GPU_NUM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$HOST_NODE_ADDR \
    train.py \
    --exp_id $EXP_ID \
    --gpu_num $GPU_NUM \
    --ddp $DDP \
    --num_cores $NUM_CORES \
    --num_epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay 1e-7 \
    --apply_lr_scheduling $LR_SCHEDULING

# Exit if training failed
if [ $? -ne 0 ]; then
    echo "Training failed. Skipping evaluation."
    exit 1
fi


# Evaluation
# python test_bev.py \
#     --exp_id $EXP_ID \
#     --gpu_num $GPU_NUM \
#     --visualization 0 \
#     --is_test_all 1


