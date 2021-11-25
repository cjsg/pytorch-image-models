#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC avtrain.py "$@"
# /home/cjsimon/anaconda3/envs/torchtest/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_PROC avtrain.py "$@"
