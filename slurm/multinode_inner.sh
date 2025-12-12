#!/bin/bash

nvidia-smi

export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1
export NCCL_SHM_DISABLE=1
export XLA_FLAGS="--xla_gpu_enable_analytical_sol_latency_estimator=false"
export HF_HOME="~/data/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python run.py --coord_addr ${MASTER_ADDR}:${MASTER_PORT} --num_procs $SLURM_NTASKS --n_layer 1
