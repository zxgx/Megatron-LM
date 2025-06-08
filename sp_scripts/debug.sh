#!/bin/bash

. run/launch.sh

WORKLOAD_NAME=debug

# where to save the log and profile files
WORKLOAD_PREFIX=$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi

# =============== program params ============
export TP_SIZE=2
export PP_SIZE=4

# =============== 1.3B model ================
export NUM_LAYER=8
export NUM_HEAD=16
export HIDDEN_SIZE=2048
export MODEL_ID=1.3B

# =================== 32k ===================
MICRO_BS=1
NUM_FOLD=2
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS))
SEQ_LEN=$((8*1024))
. local_scripts/common.sh

# torchrun --standalone --nproc_per_node $GPUS_PER_NODE $LAUNCH_1F1B_CMD

CHUNK_SIZE=4096
torchrun --standalone --nproc_per_node $GPUS_PER_NODE $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE #--ulysses-sp
    