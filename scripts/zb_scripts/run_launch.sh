echo "RUN $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"

mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/ngc-torch-23.08.sif \
    python $LAUNCH_1F1B_CMD \
    --enable-zero-bubble \
    --zero-bubble-max-pending-backward $((1*PP_SIZE)) \
    > ${WORKLOAD_PREFIX}/zb1p-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

mpirun --hostfile $HOSTFILE --np $WORLD_SIZE -N $GPUS_PER_NODE --oversubscribe \
    singularity exec --bind $SCRATCH_PATH:$SCRATCH_PATH --nv $SCRATCH_PATH/images/ngc-torch-23.08.sif \
    python $LAUNCH_1F1B_CMD \
    --enable-zero-bubble \
    --zero-bubble-max-pending-backward $((2*PP_SIZE)) \
    > ${WORKLOAD_PREFIX}/zb2p-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE}.log 2>&1

echo "FINISH $MODEL_ID model with seq: $((SEQ_LEN/1024))k, global bs: $GLOBAL_BS, micro bs: $MICRO_BS, num fold: $NUM_FOLD"
