# PoTeC_RC Task

**Task Type:** Classification  
**Number of Folds:** 4  
**Dataset:** PoTeC

## Overview

This document outlines the complete workflow for running experiments on the PoTeC_RC (reading comprehension) task.

## Workflow Steps

### 1. Data Preparation

Download and preprocess the PoTeC dataset:

```bash
tmux new-session -d -s potec 'bash src/data/preprocessing/get_data.sh PoTeC'
```

**⚠️ Important:** Verify that data was downloaded and preprocessed successfully before proceeding.

### 2. Model Checker (Cache & Validation)

Run model checker for each fold with different GPU assignments:

```bash
tmux new-session -d -s model_checker_potec_rc01 'bash run_commands/utils/model_checker.sh --data_tasks PoTeC_RC --folds 0,1 --cuda 0 --train'
tmux new-session -d -s model_checker_potec_rc23 'bash run_commands/utils/model_checker.sh --data_tasks PoTeC_RC --folds 2,3 --cuda 1 --train'

tmux new-session -d -s model_checker_potec_rc_test_dl 'bash run_commands/utils/model_checker.sh --data_tasks PoTeC_RC --cuda 1 --test'
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
bash run_commands/utils/sweep_wrapper.sh --data_tasks PoTeC_RC --folds 0,1,2,3 --wandb_project PoTeC_RC_20251118

# Create test wrapper
bash run_commands/utils/test_wrapper_creator.sh --data_task PoTeC_RC --project_name PoTeC_RC_20251118
```

### 5. Training

```bash
sbatch sweeps/PoTeC_RC_20251118/slurm/BEyeLSTMArgs/BEyeLSTMArgs_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/BEyeLSTMArgs/BEyeLSTMArgs_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/AhnRNN/AhnRNN_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/AhnRNN/AhnRNN_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/AhnCNN/AhnCNN_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/AhnCNN/AhnCNN_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/MAG/MAG_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/MAG/MAG_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PLMASArgs/PLMASArgs_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PLMASArgs/PLMASArgs_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PLMASfArgs/PLMASfArgs_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PLMASfArgs/PLMASfArgs_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/RoberteyeWord/RoberteyeWord_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/RoberteyeWord/RoberteyeWord_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/Roberta/Roberta_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/Roberta/Roberta_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/RoberteyeFixation/RoberteyeFixation_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/RoberteyeFixation/RoberteyeFixation_PoTeC_RC_folds_0_1_2_3basic.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PostFusion/PostFusion_PoTeC_RC_folds_0_1_2_3normal.job
sbatch sweeps/PoTeC_RC_20251118/slurm/PostFusion/PostFusion_PoTeC_RC_folds_0_1_2_3basic.job
```

```bash
bash sweeps/PoTeC_RC_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_PoTeC_RC_folds_0_1_2_3.sh
bash sweeps/PoTeC_RC_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_PoTeC_RC_folds_0_1_2_3.sh
bash sweeps/PoTeC_RC_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_PoTeC_RC_folds_0_1_2_3.sh
bash sweeps/PoTeC_RC_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_PoTeC_RC_folds_0_1_2_3.sh
bash sweeps/PoTeC_RC_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_PoTeC_RC_folds_0_1_2_3.sh
```

### 6. Post-Training Evaluation

```bash
# Sync outputs
bash run_commands/utils/sync_outputs_between_servers.sh

# Evaluate DL models
tmux new-session -d -s eval_potec_rc "CUDA_VISIBLE_DEVICES=1 bash sweeps/PoTeC_RC_20251118/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_potec_rc_ml 'python src/run/single_run/test_ml.py --data_task PoTeC_RC --wandb_project PoTeC_RC_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.
