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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-zxf1pkom-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/zxf1pkom; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/qid3ye3k; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/wsvjn4lx; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/acw6gken; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/zjibjoiq; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/bq6ufoe8; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/dd131xiq; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/sf0svpzj; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/auks1yc4; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_20251118/qjkzg0s9"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
