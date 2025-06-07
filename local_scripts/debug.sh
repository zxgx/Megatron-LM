#!/bin/bash

SCRATCH_PATH=exp
WORKLOAD_NAME=debug

# where to save the log and profile files
WORKLOAD_PREFIX=$SCRATCH_PATH/$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi
export PYTHONPATH=$PWD:$PYTHONPATH

echo "current py path: $PYTHONPATH"
# =============== program params ============
export TP_SIZE=2
export PP_SIZE=4

# =============== 1.3B model ================
export NUM_LAYER=8
export NUM_HEAD=16
export HIDDEN_SIZE=2048
MODEL_ID=1.3B

# =================== 32k ===================
MICRO_BS=1
NUM_FOLD=2
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS))
SEQ_LEN=$((8*1024))
. local_scripts/common.sh

torchrun --standalone --nproc_per_node 8 $LAUNCH_1F1B_CMD #\
#     # > ${WORKLOAD_PREFIX}/1f1b-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

# CHUNK_SIZE=-1
# torchrun --standalone --nproc_per_node 8 $LAUNCH_HELIX_CMD #\
    # --profile --profile-prefix $SCRATCH_PATH/tb_log
    # --chunk-size $CHUNK_SIZE \
    # > ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1
