#!/bin/bash

SCRATCH_PATH=exp
WORKLOAD_NAME=ablation_overlap

# where to save the log and profile files
export WORKLOAD_PREFIX=$SCRATCH_PATH/$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi
export PYTHONPATH=$PWD:$PYTHONPATH

# =============== program params ============
export TP_SIZE=$GPUS_PER_NODE
export PP_SIZE=$WORLD_SIZE

# =============== launch local_scripts ================
# . local_scripts/overlap_1.3b.sh
# . local_scripts/overlap_3b.sh
# . local_scripts/overlap_7b.sh
