# SBSAT_RC Task

**Task Type:** Classification  
**Number of Folds:** 4  
**Dataset:** SBSAT

## Overview

This document outlines the complete workflow for running experiments on the SBSAT_RC (Reading Comprehension) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the SBSAT dataset:

```bash
tmux new-session -d -s sbsat 'bash src/data/preprocessing/get_data.sh SBSAT'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker for training (folds parallelized):

```bash
tmux new-session -d -s model_checker_sbsat_rc0123 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_RC --folds 0,1,2,3 --cuda 0 --train'
tmux new-session -d -s model_checker_sbsat_rc23 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_RC --folds 2,3 --cuda 0 --train'
tmux new-session -d -s model_checker_sbsat_rc_test_dl1 'bash run_commands/utils/model_checker.sh --data_tasks SBSAT_RC --folds 3 --cuda 0 --test'
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
# Create sweeps
bash run_commands/utils/sweep_wrapper.sh --data_tasks SBSAT_RC --folds 0,1,2,3 --wandb_project SBSAT_RC_20251104

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task SBSAT_RC --project_name SBSAT_RC_20251104
```

### 5. Training

```bash
sbatch sweeps/SBSAT_RC_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/AhnRNN/AhnRNN_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/AhnRNN/AhnRNN_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/AhnCNN/AhnCNN_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/AhnCNN/AhnCNN_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/MAG/MAG_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/MAG/MAG_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PLMASArgs/PLMASArgs_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PLMASArgs/PLMASArgs_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PLMASfArgs/PLMASfArgs_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PLMASfArgs/PLMASfArgs_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/RoberteyeWord/RoberteyeWord_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/RoberteyeWord/RoberteyeWord_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/Roberta/Roberta_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/Roberta/Roberta_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/RoberteyeFixation/RoberteyeFixation_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/RoberteyeFixation/RoberteyeFixation_SBSAT_RC_folds_0_1_2_3basic.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PostFusion/PostFusion_SBSAT_RC_folds_0_1_2_3normal.job
sbatch sweeps/SBSAT_RC_20251104/slurm/PostFusion/PostFusion_SBSAT_RC_folds_0_1_2_3basic.job
```

```bash
bash sweeps/SBSAT_RC_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_SBSAT_RC_folds_0_1_2_3.sh
bash sweeps/SBSAT_RC_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_SBSAT_RC_folds_0_1_2_3.sh
bash sweeps/SBSAT_RC_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_SBSAT_RC_folds_0_1_2_3.sh
bash sweeps/SBSAT_RC_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_SBSAT_RC_folds_0_1_2_3.sh
bash sweeps/SBSAT_RC_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_SBSAT_RC_folds_0_1_2_3.sh
```

Train models using generated sweep scripts for local servers and DGX Slurm jobs.

### 6. Post-Training Evaluation

```bash
# Sync outputs
bash run_commands/utils/sync_outputs_between_servers.sh

# Evaluate DL models
tmux new-session -d -s eval_sbsat_rc "CUDA_VISIBLE_DEVICES=5 bash sweeps/SBSAT_RC_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_sbsat_rc_ml 'python src/run/single_run/test_ml.py --data_task SBSAT_RC --wandb_project SBSAT_RC_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
