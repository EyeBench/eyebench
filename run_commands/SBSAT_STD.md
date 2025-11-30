# SBSAT_STD Task

**Task Type:** Regression  
**Number of Folds:** 4  
**Dataset:** SBSAT

## Overview

This document outlines the complete workflow for running experiments on the SBSAT_STD (Subjective Text Difficulty) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the SBSAT dataset:

```bash
tmux new-session -d -s sbsat 'bash src/data/preprocessing/get_data.sh SBSAT'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker with regression flag (folds parallelized):

```bash
tmux new-session -d -s model_checker_sbsat_std0123 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_STD --folds 0,1,2,3 --cuda 1 --regression --train'
tmux new-session -d -s model_checker_sbsat_std23 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_STD --folds 2,3 --cuda 1 --regression --train'
tmux new-session -d -s model_checker_sbsat_std_test_dl1 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_STD --folds 0,1,2,3 --cuda 0 --regression --test'
```

**⚠️ Important:** Check `logs/failed_runs.log` to ensure no runs failed.

### 3. Data Synchronization & Cleanup

```bash
# Sync cache
bash run_commands/utils/sync_data_between_servers.sh

# Delete DEBUG results
find results/raw -type d -name "*DEBUG" -exec rm -rf {} +
```

### 4. Generate Sweep Configurations

```bash
# Create sweeps (regression mode)
bash run_commands/utils/sweep_wrapper.sh --data_tasks SBSAT_STD --folds 0,1,2,3 --wandb_project SBSAT_STD_20251104 --regression true

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task SBSAT_STD --project_name SBSAT_STD_20251104
```

### 5. Training

```bash
sbatch sweeps/SBSAT_STD_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/AhnRNN/AhnRNN_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/AhnRNN/AhnRNN_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/AhnCNN/AhnCNN_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/AhnCNN/AhnCNN_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/MAG/MAG_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/MAG/MAG_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PLMASArgs/PLMASArgs_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PLMASArgs/PLMASArgs_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PLMASfArgs/PLMASfArgs_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PLMASfArgs/PLMASfArgs_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/RoberteyeWord/RoberteyeWord_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/RoberteyeWord/RoberteyeWord_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/Roberta/Roberta_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/Roberta/Roberta_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/RoberteyeFixation/RoberteyeFixation_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/RoberteyeFixation/RoberteyeFixation_SBSAT_STD_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PostFusion/PostFusion_SBSAT_STD_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_STD_20251104/slurm/PostFusion/PostFusion_SBSAT_STD_folds_0_1_2_3basic.job
```

```bash
bash sweeps/SBSAT_STD_20251104/bash/lacc/DummyRegressorMLArgs/DummyRegressorMLArgs_SBSAT_STD_folds_0_1_2_3.sh
bash sweeps/SBSAT_STD_20251104/bash/lacc/SupportVectorRegressorMLArgs/SupportVectorRegressorMLArgs_SBSAT_STD_folds_0_1_2_3.sh
bash sweeps/SBSAT_STD_20251104/bash/lacc/LinearRegressionArgs/LinearRegressionArgs_SBSAT_STD_folds_0_1_2_3.sh
bash sweeps/SBSAT_STD_20251104/bash/lacc/LinearMeziereArgs/LinearMeziereArgs_SBSAT_STD_folds_0_1_2_3.sh
bash sweeps/SBSAT_STD_20251104/bash/lacc/RandomForestRegressorMLArgs/RandomForestRegressorMLArgs_SBSAT_STD_folds_0_1_2_3.sh
```

Train models using generated sweep scripts for local servers and DGX Slurm jobs.

### 6. Post-Training Evaluation

```bash
# Sync outputs
bash run_commands/utils/sync_outputs_between_servers.sh

# Evaluate DL models
tmux new-session -d -s eval_sbsat_std "CUDA_VISIBLE_DEVICES=7 bash sweeps/SBSAT_STD_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_sbsat_std_ml 'python src/run/single_run/test_ml.py --data_task SBSAT_STD --wandb_project SBSAT_STD_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
