# CopCo_RCS Task

**Task Type:** Regression  
**Number of Folds:** 4  
**Dataset:** CopCo

## Overview

This document outlines the complete workflow for running experiments on the CopCo_RCS (Reading Comprehension Skill) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the CopCo dataset:

```bash
tmux new-session -d -s copco 'bash src/data/preprocessing/get_data.sh CopCo'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker to:
- Save cache of preprocessed data
- Ensure no failed runs

```bash
# Training validation
tmux new-session -d -s model_checker_copco_rcs0123 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_RCS --folds 0,1,2,3 --cuda 0 --regression --train'
tmux new-session -d -s model_checker_copco_rcs1 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_RCS --folds 1 --cuda 1 --regression --train'
tmux new-session -d -s model_checker_copco_rcs2 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_RCS --folds 2 --cuda 0 --regression --train'
tmux new-session -d -s model_checker_copco_rcs3 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_RCS --folds 3 --cuda 1 --regression --train'

# Testing validation
tmux new-session -d -s model_checker_copco_rcs_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_RCS --folds 0,1,2,3 --cuda 1 --regression --test'
```

**⚠️ Important:** Check `logs/failed_runs.log` to ensure no runs failed.

### 3. Data Synchronization & Cleanup

Sync cached data across servers and clean up debug results:

```bash
# Sync cache between servers
bash run_commands/utils/sync_data_between_servers.sh

# Delete DEBUG results
find results/raw -type d -name "*DEBUG" -exec rm -rf {} +
```

### 4. Generate Sweep Configurations

Create hyperparameter sweep configurations and test wrappers:

```bash
# Create sweeps
bash run_commands/utils/sweep_wrapper.sh --data_tasks CopCo_RCS --folds 0,1,2,3 --wandb_project CopCo_RCS_20251104 --regression true

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task CopCo_RCS --project_name CopCo_RCS_20251104
```

### 5. Training

```bash
sbatch sweeps/CopCo_RCS_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/AhnRNN/AhnRNN_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/AhnRNN/AhnRNN_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/AhnCNN/AhnCNN_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/AhnCNN/AhnCNN_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/MAG/MAG_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/MAG/MAG_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/Roberta/Roberta_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/Roberta/Roberta_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_RCS_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PostFusion/PostFusion_CopCo_RCS_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_RCS_20251104/slurm/PostFusion/PostFusion_CopCo_RCS_folds_0_1_2_3basic.job
```

```bash
bash sweeps/CopCo_RCS_20251104/bash/lacc/DummyRegressorMLArgs/DummyRegressorMLArgs_CopCo_RCS_folds_0_1_2_3.sh
bash sweeps/CopCo_RCS_20251104/bash/lacc/SupportVectorRegressorMLArgs/SupportVectorRegressorMLArgs_CopCo_RCS_folds_0_1_2_3.sh
bash sweeps/CopCo_RCS_20251104/bash/lacc/LinearRegressionArgs/LinearRegressionArgs_CopCo_RCS_folds_0_1_2_3.sh
bash sweeps/CopCo_RCS_20251104/bash/lacc/LinearMeziereArgs/LinearMeziereArgs_CopCo_RCS_folds_0_1_2_3.sh
bash sweeps/CopCo_RCS_20251104/bash/lacc/RandomForestRegressorMLArgs/RandomForestRegressorMLArgs_CopCo_RCS_folds_0_1_2_3.sh
```

### 6. Post-Training Evaluation

After training completes, synchronize outputs and run evaluation:

```bash
# Sync outputs between servers
tmux new-session -d -s sync_output "bash run_commands/utils/sync_outputs_between_servers.sh" ; tmux set-option remain-on-exit off
tmux new-session -d -s sync_output_dgx "bash run_commands/utils/sync_outputs_between_servers_dgx.sh" ; tmux set-option remain-on-exit off

# Evaluate Deep Learning models
tmux new-session -d -s eval_copco_rcs "CUDA_VISIBLE_DEVICES=1 bash sweeps/CopCo_RCS_20251104/test_dl_wrapper.sh"

# Evaluate Machine Learning models
tmux new-session -d -s eval_copco_rcs_ml 'python src/run/single_run/test_ml.py --data_task CopCo_RCS --wandb_project CopCo_RCS_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub after completion.
