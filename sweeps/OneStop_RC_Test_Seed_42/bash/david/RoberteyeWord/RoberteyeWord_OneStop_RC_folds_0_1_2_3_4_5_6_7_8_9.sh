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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-qjn5ye19-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/qjn5ye19; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/4m8qwrie; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/p2upmm0k; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/hf9txzuo; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/goqz33x2; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/o0zyogdd; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/oppavwzo; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/ewqagawm; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/q1gx5562; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_42/1vhmggts"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
