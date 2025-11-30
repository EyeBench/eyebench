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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-2p8dgxu4-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/2p8dgxu4; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/pnovzjzh; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/o8gnhjx6; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/sko955oo; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/j0pab37w; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/ia9mq6g3; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/lqcdnqa0; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/uvl0v4ih; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/ixjtr9wn; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStop_RC_Test_Seed_41/apaqejtg"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
