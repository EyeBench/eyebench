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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-skq7r8o8-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/skq7r8o8; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/2sc1epsk; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/4gp92lnd; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/xh5yrr1a; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/3rqadcvr; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/qway8a8j; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/1bon2l6z; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/he1r4v9t; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/ay9inr4l; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251104/8rbi568p"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
