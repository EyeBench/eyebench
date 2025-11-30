#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh
cd /mnt/mlshare/reich3/eyebench_private

GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-bodgp9f6-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/bodgp9f6; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/r0yq1zvp; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/0t50815l; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/1e0vzo1u"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
