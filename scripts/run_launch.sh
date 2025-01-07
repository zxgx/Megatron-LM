echo "RUN $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"

# echo "output: ${WORKLOAD_PREFIX}/1f1b-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}.log"
# echo "output: ${WORKLOAD_PREFIX}/helix-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-<chunk-size>-tp${TP_SIZE}-pp${PP_SIZE}.log"

mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/pytorch_23.08-py3.sif \
    python $LAUNCH_1F1B_CMD \
    > ${WORKLOAD_PREFIX}/1f1b-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

CHUNK_SIZE=-1
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/pytorch_23.08-py3.sif \
    python $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    > ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

CHUNK_SIZE=2048
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/pytorch_23.08-py3.sif \
    python $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    > ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

CHUNK_SIZE=4096
mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/pytorch_23.08-py3.sif \
    python $LAUNCH_HELIX_CMD \
    --chunk-size $CHUNK_SIZE \
    > ${WORKLOAD_PREFIX}/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

echo "FINISH $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"
