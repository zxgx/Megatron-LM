#!/bin/bash

SCRATCH_PATH=exp
WORKLOAD_NAME=trace

# where to save the log and profile files
WORKLOAD_PREFIX=$SCRATCH_PATH/$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi
export PYTHONPATH=$PWD:$PYTHONPATH

echo "current py path: $PYTHONPATH"
# =============== program params ============
export TP_SIZE=$GPUS_PER_NODE
export PP_SIZE=$WORLD_SIZE

# =============== 1.3B model ================
export NUM_LAYER=8
export NUM_HEAD=32
export HIDDEN_SIZE=4096
MODEL_ID=1.3B

# =================== 128k ===================
MICRO_BS=1
NUM_FOLD=2
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS))
SEQ_LEN=$((32*1024))
. local_scripts/common.sh

CHUNK_SIZE=4096
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    --profile --profile-prefix $WORKLOAD_PREFIX/PP${PP_SIZE}_trace_seq$((SEQ_LEN/1024))
