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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-xkhtdguw-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_TYP_20251104/xkhtdguw; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_TYP_20251104/bkd5b17a; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_TYP_20251104/wghp9o6x; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_TYP_20251104/go3ytibj"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
