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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-47v0eqz3-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/IITBHGC_CV_20251104/47v0eqz3; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/IITBHGC_CV_20251104/zhbiwf6c; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/IITBHGC_CV_20251104/g6xg2w2q; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/IITBHGC_CV_20251104/zh01433x"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
