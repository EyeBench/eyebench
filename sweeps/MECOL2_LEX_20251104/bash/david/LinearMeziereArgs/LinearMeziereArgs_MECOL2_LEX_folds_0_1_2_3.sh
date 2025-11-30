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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-1rvxsj3s-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/MECOL2_LEX_20251104/1rvxsj3s; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/MECOL2_LEX_20251104/z3pif4ph; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/MECOL2_LEX_20251104/k7mpmo0l; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/MECOL2_LEX_20251104/zdblcgle"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
