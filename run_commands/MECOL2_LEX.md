# MECOL2_LEX Task

**Task Type:** Regression  
**Number of Folds:** 4  
**Dataset:** MECOL2

## Overview

This document outlines the complete workflow for running experiments on the MECOL2_LEX (Lexical Knowledge) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the MECOL2 dataset:

```bash
tmux new-session -d -s mecol2 'bash src/data/preprocessing/get_data.sh MECOL2'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker for each fold (regression task):

```bash
tmux new-session -d -s model_checker_mecol2_lex0123 'bash run_commands/utils/model_checker.sh --data_tasks MECOL2_LEX --folds 0,1,2,3 --cuda 1 --regression --train'
tmux new-session -d -s model_checker_mecol2_lex1 'bash run_commands/utils/model_checker.sh --data_tasks MECOL2_LEX --folds 1 --cuda 5 --regression --train'
tmux new-session -d -s model_checker_mecol2_lex2 'bash run_commands/utils/model_checker.sh --data_tasks MECOL2_LEX --folds 2 --cuda 6 --regression --train'
tmux new-session -d -s model_checker_mecol2_lex3 'bash run_commands/utils/model_checker.sh --data_tasks MECOL2_LEX --folds 3 --cuda 7 --regression --train'

tmux new-session -d -s model_checker_mecol2_lex_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks MECOL2_LEX --folds 0,1,2,3 --cuda 0 --regression --test'
```

**⚠️ Important:** Check `logs/failed_runs.log` to ensure no runs failed.

### 3. Data Synchronization & Cleanup

```bash
# Sync cache (in tmux session)
tmux new-session -d -s sync_data2 'bash run_commands/utils/sync_data_between_servers.sh'

# Delete DEBUG results
find results/raw -type d -name "*DEBUG" -exec rm -rf {} +
```

### 4. Generate Sweep Configurations

```bash
# Create sweeps (regression mode)
bash run_commands/utils/sweep_wrapper.sh --data_tasks MECOL2_LEX --folds 0,1,2,3 --wandb_project MECOL2_LEX_20251104 --regression true

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task MECOL2_LEX --project_name MECOL2_LEX_20251104
```

### 5. Training

```bash
sbatch sweeps/MECOL2_LEX_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/AhnRNN/AhnRNN_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/AhnRNN/AhnRNN_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/AhnCNN/AhnCNN_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/AhnCNN/AhnCNN_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/MAG/MAG_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/MAG/MAG_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PLMASArgs/PLMASArgs_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PLMASArgs/PLMASArgs_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PLMASfArgs/PLMASfArgs_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PLMASfArgs/PLMASfArgs_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/RoberteyeWord/RoberteyeWord_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/RoberteyeWord/RoberteyeWord_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/Roberta/Roberta_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/Roberta/Roberta_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/RoberteyeFixation/RoberteyeFixation_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/RoberteyeFixation/RoberteyeFixation_MECOL2_LEX_folds_0_1_2_3basic.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PostFusion/PostFusion_MECOL2_LEX_folds_0_1_2_3normal.job
sbatch sweeps/MECOL2_LEX_20251104/slurm/PostFusion/PostFusion_MECOL2_LEX_folds_0_1_2_3basic.job
```

```bash
bash sweeps/MECOL2_LEX_20251104/bash/lacc/DummyRegressorMLArgs/DummyRegressorMLArgs_MECOL2_LEX_folds_0_1_2_3.sh
bash sweeps/MECOL2_LEX_20251104/bash/lacc/SupportVectorRegressorMLArgs/SupportVectorRegressorMLArgs_MECOL2_LEX_folds_0_1_2_3.sh
bash sweeps/MECOL2_LEX_20251104/bash/lacc/LinearRegressionArgs/LinearRegressionArgs_MECOL2_LEX_folds_0_1_2_3.sh
bash sweeps/MECOL2_LEX_20251104/bash/lacc/LinearMeziereArgs/LinearMeziereArgs_MECOL2_LEX_folds_0_1_2_3.sh
bash sweeps/MECOL2_LEX_20251104/bash/lacc/RandomForestRegressorMLArgs/RandomForestRegressorMLArgs_MECOL2_LEX_folds_0_1_2_3.sh
```

Train models using generated sweep scripts for both small and large servers, plus DGX Slurm jobs.


### 6. Post-Training Evaluation

```bash
# Sync outputs
tmux new-session -d -s sync_output "bash run_commands/utils/sync_outputs_between_servers.sh" ; tmux set-option remain-on-exit off
tmux new-session -d -s sync_output_dgx "bash run_commands/utils/sync_outputs_between_servers_dgx.sh" ; tmux set-option remain-on-exit off

# Evaluate DL models
tmux new-session -d -s eval_mecol2_lex "CUDA_VISIBLE_DEVICES=0 bash sweeps/MECOL2_LEX_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_mecol2_lex_ml 'python src/run/single_run/test_ml.py --data_task MECOL2_LEX --wandb_project MECOL2_LEX_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
