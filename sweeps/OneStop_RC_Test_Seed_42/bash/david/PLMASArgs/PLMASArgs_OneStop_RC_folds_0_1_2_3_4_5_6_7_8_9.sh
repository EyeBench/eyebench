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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-j2pj3zj7-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/j2pj3zj7; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/9myzt0x1; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/tuu191n1; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/inbvesxf; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/541zqkml; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/ioyy1fym; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/utu2qpvm; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/vwylp884; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/2ctm23uq; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/40kn9u2n"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
