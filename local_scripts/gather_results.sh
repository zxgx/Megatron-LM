#!/bin/bash

TP_SIZE=8
NUM_FOLD=2
MICRO_BS=1

for PP_SIZE in 2 4 8; do
    GLOBAL_BS=$((2*PP_SIZE*MICRO_BS))

    for MODEL_ID in 1.3B 3B 7B; do
        for SEQ_LEN in $((32*1024)) $((64*1024)) $((96*1024)) $((128*1024)); do
            for log_dir in exp exp_h20; do
            # for log_dir in exp; do # use this line if only one experiment platform
                echo processing PP $PP_SIZE model ${MODEL_ID} seq len $SEQ_LEN experiment dir $log_dir
                # 1F1B
                python local_scripts/extract2.py \
                    --log-dir $log_dir/end2end/1f1b-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE} \
                    --pp-size $PP_SIZE --output-dir $log_dir/PP${PP_SIZE}_results
                
                # ZB1P
                python local_scripts/extract2.py \
                    --log-dir $log_dir/zerobubble_end2end/zb1p-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE} \
                    --pp-size $PP_SIZE --output-dir $log_dir/PP${PP_SIZE}_results
                
                # AdaPipe
                python local_scripts/extract2.py \
                    --log-dir $log_dir/adapipe_end2end/adapipe-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-tp${TP_SIZE}-pp${PP_SIZE} \
                    --pp-size $PP_SIZE --output-dir $log_dir/PP${PP_SIZE}_results

                for CHUNK_SIZE in -1 2048 4096; do
                    python local_scripts/extract2.py \
                        --log-dir $log_dir/end2end/helix-$MODEL_ID-f${NUM_FOLD}-$((SEQ_LEN/1024))k-B${GLOBAL_BS}-b${MICRO_BS}-${CHUNK_SIZE}-tp${TP_SIZE}-pp${PP_SIZE} \
                        --pp-size $PP_SIZE --output-dir $log_dir/PP${PP_SIZE}_results
                done
            done
        done
    done
done