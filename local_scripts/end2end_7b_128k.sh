#!/bin/bash

# experiment config
. local_scripts/exp_end2end.sh

# model config
. local_scripts/run_7b.sh

# training config
MICRO_BS=1
NUM_FOLD=2
SEQ_LEN=$((128*1024))
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS))
. local_scripts/common.sh

# launch scripts
echo "RUN $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"

torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $LAUNCH_1F1B_CMD \
    | tee  ${WORKLOAD_PREFIX}/1f1b-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}-${RANK}-${WORLD_SIZE}.log

CHUNK_SIZE=-1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    | tee  ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-${RANK}-${WORLD_SIZE}.log

CHUNK_SIZE=2048
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    | tee  ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-${RANK}-${WORLD_SIZE}.log

CHUNK_SIZE=4096
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    | tee  ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-${RANK}-${WORLD_SIZE}.log

echo "FINISH $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"

