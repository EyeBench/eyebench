# IITBHGC_CV Task

**Task Type:** Classification  
**Number of Folds:** 4  
**Dataset:** IITBHGC

## Overview

This document outlines the complete workflow for running experiments on the IITBHGC_CV (claim verification) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the IITBHGC dataset:

```bash
tmux new-session -d -s iitbhgc 'bash src/data/preprocessing/get_data.sh IITBHGC'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker for each fold separately:

```bash
tmux new-session -d -s model_checker_iitbhgc_cv0 'bash run_commands/utils/model_checker.sh --data_tasks IITBHGC_CV --folds 0 --cuda 0 --train'
tmux new-session -d -s model_checker_iitbhgc_cv1 'bash run_commands/utils/model_checker.sh --data_tasks IITBHGC_CV --folds 1 --cuda 1 --train'
tmux new-session -d -s model_checker_iitbhgc_cv2 'bash run_commands/utils/model_checker.sh --data_tasks IITBHGC_CV --folds 2 --cuda 0 --train'
tmux new-session -d -s model_checker_iitbhgc_cv3 'bash run_commands/utils/model_checker.sh --data_tasks IITBHGC_CV --folds 3 --cuda 1 --train'

tmux new-session -d -s model_checker_iitbhgc_cv_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks IITBHGC_CV --cuda 1 --test'
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
bash run_commands/utils/sweep_wrapper.sh --data_tasks IITBHGC_CV --folds 0,1,2,3 --wandb_project IITBHGC_CV_20251104

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task IITBHGC_CV --project_name IITBHGC_CV_20251104
```

### 5. Training

```bash

sbatch sweeps/IITBHGC_CV_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/AhnRNN/AhnRNN_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/AhnRNN/AhnRNN_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/AhnCNN/AhnCNN_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/AhnCNN/AhnCNN_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/MAG/MAG_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/MAG/MAG_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PLMASArgs/PLMASArgs_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PLMASArgs/PLMASArgs_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PLMASfArgs/PLMASfArgs_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PLMASfArgs/PLMASfArgs_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/RoberteyeWord/RoberteyeWord_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/RoberteyeWord/RoberteyeWord_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/Roberta/Roberta_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/Roberta/Roberta_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/RoberteyeFixation/RoberteyeFixation_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/RoberteyeFixation/RoberteyeFixation_IITBHGC_CV_folds_0_1_2_3basic.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PostFusion/PostFusion_IITBHGC_CV_folds_0_1_2_3normal.job
sbatch sweeps/IITBHGC_CV_20251104/slurm/PostFusion/PostFusion_IITBHGC_CV_folds_0_1_2_3basic.job
```

```bash
bash sweeps/IITBHGC_CV_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_IITBHGC_CV_folds_0_1_2_3.sh
bash sweeps/IITBHGC_CV_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_IITBHGC_CV_folds_0_1_2_3.sh
bash sweeps/IITBHGC_CV_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_IITBHGC_CV_folds_0_1_2_3.sh
bash sweeps/IITBHGC_CV_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_IITBHGC_CV_folds_0_1_2_3.sh
bash sweeps/IITBHGC_CV_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_IITBHGC_CV_folds_0_1_2_3.sh
```

Training scripts are available for:

- **LACC servers** (local computation cluster)
- **DAVID server** (alternative compute server)
- **DGX** (via Slurm)

### 6. Post-Training Evaluation

```bash
# Sync outputs
bash run_commands/utils/sync_outputs_between_servers.sh

# Evaluate DL models
tmux new-session -d -s eval_iitbhgc_cv "CUDA_VISIBLE_DEVICES=3 bash sweeps/IITBHGC_CV_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_iitbhgc_cv_ml "python src/run/single_run/test_ml.py --data_task IITBHGC_CV --wandb_project IITBHGC_CV_20251104"
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
