# =============== 1.3B model ================
export NUM_LAYER=24
export NUM_HEAD=16
export HIDDEN_SIZE=2048
MODEL_ID=1.3B

# =================== 32k ===================
MICRO_BS=1
NUM_FOLD=1
GLOBAL_BS=$((PP_SIZE*NUM_FOLD*MICRO_BS*2))
SEQ_LEN=$((32*1024))
. scripts/common.sh

. scripts/run_launch.sh

# =================== 64k ===================
SEQ_LEN=$((64*1024))
. scripts/common.sh

. scripts/run_launch.sh

# =================== 96k ===================
SEQ_LEN=$((96*1024))
. scripts/common.sh

. scripts/run_launch.sh

# =================== 128k ===================
SEQ_LEN=$((128*1024))
. scripts/common.sh

. scripts/run_launch.sh

# =================== 256k ===================
# SEQ_LEN=$((256*1024))
# . scripts/common.sh

# . scripts/overlap_launch.sh
