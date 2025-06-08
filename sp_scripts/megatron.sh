#!/bin/bash

. run/launch.sh

WORKLOAD_NAME=sp_log

# where to save the log and profile files
WORKLOAD_PREFIX=$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi

# =============== program params ============
export TP_SIZE=4
export PP_SIZE=2

# =============== 1.3B model ================
export NUM_LAYER=24
export NUM_HEAD=16
export HIDDEN_SIZE=2048
export MODEL_ID=1.3B

# =================== 32k ===================
MICRO_BS=1
NUM_FOLD=2
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS))
# SEQ_LEN=$((128*1024))
# . local_scripts/common.sh


CHUNK_SIZE=4096

for seq_base in 32 64 96 128; do
    SEQ_LEN=$((seq_base*1024))
    . local_scripts/common.sh
    
    torchrun --standalone --nproc_per_node $GPUS_PER_NODE $LAUNCH_HELIX_CMD \
        --chunk-size $CHUNK_SIZE | tee $WORKLOAD_NAME/megatron_${seq_base}k.log

    torchrun --standalone --nproc_per_node $GPUS_PER_NODE $LAUNCH_HELIX_CMD \
        --chunk-size $CHUNK_SIZE --ulysses-sp | tee $WORKLOAD_NAME/ulysses_${seq_base}k.log

done
