#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
cd $HOME/eyebench_private

GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-3ibql7vx-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/3ibql7vx; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/3g6582kt; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/4p0d4lih; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/SBSAT_RC_20251104/xz97bkie"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
