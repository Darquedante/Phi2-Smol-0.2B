#!/usr/bin/bash
# This script accelerates the launch of a process with specific options.

# Usage: Provide a training script as an argument when running this script, e.g., ./train.sh sft.py

accelerate launch --multi_gpu --num_processes 2 --config_file accelerate_multi_gpu.yaml $1
