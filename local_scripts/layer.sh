#!/bin/bash

# =============== env params ================
# where to save the log and profile files
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

# =============== program params ============
export DP_SIZE=1
export TP_SIZE=8
export PP_SIZE=1

export MICRO_BS=1
export GLOBAL_BS=1

# =============== model params ================
export NUM_LAYER=24
export NUM_HEAD=32
export HIDDEN_SIZE=4096

# =============== launch cmd ================
WORKLOAD_PREFIX=./layer_profile
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir "$WORKLOAD_PREFIX"
    echo "CREATE: $WORKLOAD_PREFIX"
fi

for ((seq_exp=2; seq_exp<=2; seq_exp++)); do
    seq_base=$((2**$seq_exp))
    SEQ_LEN=$(($seq_base*1024))
    NUM_HEAD=$(($HIDDEN_SIZE/128))
    . local_scripts/common.sh
    
    torchrun $DISTRIBUTED_ARGS local_scripts/benchmark/transformer_layer.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --recompute-activations  \
        --use-flash-attn \
        --no-attention-mask \
        --use-triton-fusion \
        --parallel-position-embedding \
        --profile-prefix ${WORKLOAD_PREFIX}/trace_${seq_base}k
    
    echo "FINISH micro bs: ${MICRO_BS}, seq: ${seq_base}k, hidden size: ${HIDDEN_SIZE}"
done

