#!/bin/bash

A=$1

set --

export XLA_FLAGS="--xla_gpu_enable_analytical_sol_latency_estimator=false"
export HF_HOME="~/data/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
source ~/data/miniforge3/bin/activate
conda activate nanoegg
cd ~/data/nano-egg

python run.py --parallel_generations_per_gpu $A --track --validate_every 10 --group_size 2
