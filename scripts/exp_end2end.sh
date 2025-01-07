#!/bin/bash
#PBS -P H100008
#PBS -l select=4:ngpus=8
#PBS -l place=vscatter
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o exp_end2end.log

# =============== env params ================
# where the ngc docker image is saved
SCRATCH_PATH=$HPCTMP
WORKLOAD_NAME=exp_end2end-noseg

# where to save the log and profile files
WORKLOAD_PREFIX=$SCRATCH_PATH/archive/$WORKLOAD_NAME
if [ ! -d "$WORKLOAD_PREFIX" ]; then
    mkdir -p $WORKLOAD_PREFIX
    echo "CREATE: $WORKLOAD_PREFIX"
fi

# $TMPDIR set by PBS will intervene triton compilation inside singularity
export RECORD=$TMPDIR
unset TMPDIR
echo "JOB TMPDIR: $TMPDIR, record tmpdir: $RECORD"

cd $PBS_O_WORKDIR
echo "JOB ID: $PBS_JOBID, pwd: $PWD, pbs workdir: $PBS_O_WORKDIR"

export NNODES=4
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | awk -F'.' '{print $1}')
export MASTER_PORT=29502
echo "master node: $MASTER_ADDR"

export HOSTFILE="$PBS_JOBID.hostfile"
cat $PBS_NODEFILE | awk -F'.' '{for(i=1;i<=NF;i+=6) print $1 " slots="ENVIRON["GPUS_PER_NODE"]}' > $HOSTFILE
echo "detected hosts: $(cat $HOSTFILE)"

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(printf "%s," $(seq 0 $(($GPUS_PER_NODE-1))) | sed 's/,$//')
echo "singularity cuda visible devices: $SINGULARITYENV_CUDA_VISIBLE_DEVICES"

# =============== program params ============
export TP_SIZE=$GPUS_PER_NODE
export PP_SIZE=$NNODES

# =============== launch scripts ================
. scripts/run_1.3b.sh
. scripts/run_3b.sh
. scripts/run_7b.sh

rm $HOSTFILE
