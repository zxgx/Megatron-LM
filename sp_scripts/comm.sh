#!/bin/bash

. run/launch.sh

torchrun --standalone --nproc_per_node $GPUS_PER_NODE local_scripts/benchmark/communication.py
