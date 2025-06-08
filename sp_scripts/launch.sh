#!/bin/bash

# =============== program params ================
export PYTHONPATH=$PYTHONPATH:$PWD

# export NCCL_IB_DISABLE=0
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_11:1

export NNODES=1
export NODE_RANK=0
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
export MASTER_ADDR=10.42.178.107
export MASTER_PORT=9527
