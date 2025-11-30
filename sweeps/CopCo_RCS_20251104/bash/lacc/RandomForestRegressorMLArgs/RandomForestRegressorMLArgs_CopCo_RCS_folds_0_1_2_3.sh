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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-77j7vbck-4"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_RCS_20251104/77j7vbck; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_RCS_20251104/snvaxgeh; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_RCS_20251104/ss4n0gh1; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/CopCo_RCS_20251104/6zqq5u8e"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
