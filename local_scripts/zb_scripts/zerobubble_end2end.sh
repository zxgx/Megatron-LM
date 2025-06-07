#!/bin/bash

SCRATCH_PATH=exp
WORKLOAD_NAME=zerobubble_end2end

# where to save the log and profile files
WORKLOAD_PREFIX=$SCRATCH_PATH/$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi

# =============== program params ============
export TP_SIZE=$GPUS_PER_NODE
export PP_SIZE=$WORLD_SIZE

# =============== launch scripts ================
. scripts/run_1.3b.sh
. scripts/run_3b.sh
. scripts/run_7b.sh
