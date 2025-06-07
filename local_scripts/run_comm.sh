#!/bin/bash

export GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NNODES=1
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $ISTRIBUTED_ARGS local_scripts/benchmark/communication.py
