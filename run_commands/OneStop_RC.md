# OneStop_RC Task

**Task Type:** Classification  
**Number of Folds:** 10  
**Dataset:** OneStop

## Overview

This document outlines the complete workflow for running experiments on the OneStop_RC (Reading Comprehension) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the OneStop dataset:

```bash
tmux new-session -d -s data_onestop 'bash src/data/preprocessing/get_data.sh OneStop'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker across all 10 folds (parallelized across GPUs):

```bash
tmux new-session -d -s model_checker_onestop_rc01 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --folds 0,1 --cuda 0 --train'
tmux new-session -d -s model_checker_onestop_rc23 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --folds 2,3 --cuda 1 --train'
tmux new-session -d -s model_checker_onestop_rc45 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --folds 4,5 --cuda 0 --train'
tmux new-session -d -s model_checker_onestop_rc67 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --folds 6,7 --cuda 0 --train'
tmux new-session -d -s model_checker_onestop_rc89 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --folds 8,9 --cuda 1 --train'

tmux new-session -d -s model_checker_onestop_rc_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks OneStop_RC --cuda 1 --test'
```

**⚠️ Important:** Check `logs/failed_runs.log` or run `python logs/parse_to_csv.py` to verify no failed runs.

### 3. Data Synchronization & Cleanup

```bash
# Sync cache
bash run_commands/utils/sync_data_between_servers.sh
bash run_commands/utils/sync_data_to_dgx.sh


# Delete DEBUG results
find results/raw -type d -name "*DEBUG" -exec rm -rf {} +
```

### 4. Generate Sweep Configurations

```bash
# Create sweeps
bash run_commands/utils/sweep_wrapper.sh --data_tasks OneStop_RC --folds 0,1,2,3,4,5,6,7,8,9 --wandb_project OneStop_RC_20251118

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task OneStop_RC --project_name OneStop_RC_20251118
```

### 5. Training

```bash
sbatch sweeps/OneStop_RC_20251118/slurm/BEyeLSTMArgs/BEyeLSTMArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/BEyeLSTMArgs/BEyeLSTMArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/AhnRNN/AhnRNN_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/AhnRNN/AhnRNN_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/AhnCNN/AhnCNN_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/AhnCNN/AhnCNN_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/MAG/MAG_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/MAG/MAG_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/PLMASArgs/PLMASArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/PLMASArgs/PLMASArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/PLMASfArgs/PLMASfArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/PLMASfArgs/PLMASfArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/RoberteyeWord/RoberteyeWord_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/RoberteyeWord/RoberteyeWord_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/Roberta/Roberta_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/Roberta/Roberta_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/RoberteyeFixation/RoberteyeFixation_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/RoberteyeFixation/RoberteyeFixation_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
sbatch sweeps/OneStop_RC_20251118/slurm/PostFusion/PostFusion_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9normal.job
sbatch sweeps/OneStop_RC_20251118/slurm/PostFusion/PostFusion_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9basic.job
```

```bash
bash sweeps/OneStop_RC_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9.sh
bash sweeps/OneStop_RC_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9.sh
bash sweeps/OneStop_RC_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9.sh
bash sweeps/OneStop_RC_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9.sh
bash sweeps/OneStop_RC_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_OneStop_RC_folds_0_1_2_3_4_5_6_7_8_9.sh
```

### 6. Post-Training Evaluation

```bash
# Sync outputs from DGX
tmux new-session -d -s rsync_dgx 'bash run_commands/utils/sync_outputs_between_servers_dgx.sh'

# After training is done:
bash run_commands/utils/sync_outputs_between_servers.sh

# Evaluate DL models
tmux new-session -d -s eval_onestop_rc_dl "CUDA_VISIBLE_DEVICES=0 bash sweeps/OneStop_RC_20251118/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_onestop_rc_ml "python src/run/single_run/test_ml.py --data_task OneStop_RC --wandb_project OneStop_RC_20251104"
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
