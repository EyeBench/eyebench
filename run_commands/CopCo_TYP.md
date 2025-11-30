# CopCo_TYP Task

**Task Type:** Classification  
**Number of Folds:** 4  
**Dataset:** CopCo

## Overview

This document outlines the complete workflow for running experiments on the CopCo_TYP (Typicality Classification) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the CopCo dataset:

```bash
tmux new-session -d -s copco 'bash src/data/preprocessing/get_data.sh CopCo'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker to save cache of preprocessed data and ensure no failed runs:

```bash
# Training validation
tmux new-session -d -s model_checker_copco_typ0123 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --cuda 0 --train'
tmux new-session -d -s model_checker_copco_typ1 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP --folds 1 --cuda 1 --train'
tmux new-session -d -s model_checker_copco_typ2 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP --folds 2 --cuda 0 --train'
tmux new-session -d -s model_checker_copco_typ3 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP --folds 3 --cuda 1 --train'

# Testing validation
tmux new-session -d -s model_checker_copco_typ_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --cuda 1 --test'
```

**⚠️ Important:** Check `logs/failed_runs.log` to ensure no runs failed.

### 3. Data Synchronization & Cleanup

```bash
# Sync cache between servers
bash run_commands/utils/sync_data_between_servers.sh

# Delete DEBUG results
find results/raw -type d -name "*DEBUG" -exec rm -rf {} +
```

### 4. Generate Sweep Configurations

```bash
# Create sweeps
bash run_commands/utils/sweep_wrapper.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --wandb_project CopCo_TYP_20251104

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task CopCo_TYP --project_name CopCo_TYP_20251104
```

### 5. Training

#### Deep Learning Models

Run training scripts for DL models (GPU assignments specified per model).

```bash

sbatch sweeps/CopCo_TYP_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/BEyeLSTMArgs/BEyeLSTMArgs_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/AhnRNN/AhnRNN_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/AhnRNN/AhnRNN_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/AhnCNN/AhnCNN_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/AhnCNN/AhnCNN_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/MAG/MAG_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/MAG/MAG_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/Roberta/Roberta_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/Roberta/Roberta_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PostFusion/PostFusion_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PostFusion/PostFusion_CopCo_TYP_folds_0_1_2_3basic.job
```

```bash
bash sweeps/CopCo_TYP_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_CopCo_TYP_folds_0_1_2_3.sh
```

### 6. Post-Training Evaluation

Run on laccl-srv1 after training is complete:

```bash
# Sync outputs
tmux new-session -d -s sync_output "bash run_commands/utils/sync_outputs_between_servers.sh" ; tmux set-option remain-on-exit off
tmux new-session -d -s sync_output_dgx "bash run_commands/utils/sync_outputs_between_servers_dgx.sh" ; tmux set-option remain-on-exit off

# Evaluate DL models
tmux new-session -d -s eval_copco_typ "CUDA_VISIBLE_DEVICES=2 bash sweeps/CopCo_TYP_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_copco_typ_ml 'python src/run/single_run/test_ml.py --data_task CopCo_TYP --wandb_project CopCo_TYP_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
