#!/bin/bash

# model_checker.sh
# This script runs multiple model/data/trainer combinations for specified data tasks and folds.
#
# Usage:
#   ./model_checker.sh [--data_tasks task1,task2,...] [--folds fold1,fold2,...] [--cuda device_ids] [--regression true|false] [--train true|false] [--test true|false]
#
# Arguments:
#   --data_tasks   Comma-separated list of data tasks to run (default: CopCo_TYP)
#   --folds        Comma-separated list of fold indices to run (default: 0)
#   --cuda         CUDA device IDs to use (default: 0)
#   --regression   Whether to use regression models (default: false)
#   --train        Whether to run training (default: true)
#   --test         Whether to run testing (default: false)
#
# Example:
#   bash run_commands/utils/model_checker.sh --data_tasks CopCo_TYP,AnotherTask --folds 0,1 --cuda 0,1 --regression true --train true --test true

# Parse arguments
regression="false"
train="false"
test="false"

while [[ $# -gt 0 ]]; do
    case $1 in
    --data_tasks)
        IFS=',' read -ra data_tasks <<<"$2"
        shift 2
        ;;
    --folds)
        IFS=',' read -ra folds <<<"$2"
        shift 2
        ;;
    --cuda)
        CUDA_VISIBLE_DEVICES="$2"
        shift 2
        ;;
    --regression)
        regression="true"
        shift
        ;;
    --train)
        train="true"
        shift
        ;;
    --test)
        test="true"
        shift
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done


echo "Data tasks: ${data_tasks[@]}"
echo "Folds: ${folds[@]}"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Regression: $regression"
echo "Train: $train"
echo "Test: $test"
overwrite_data="True" # Set to true to overwrite existing data
echo "Overwrite data: $overwrite_data"

# Always include DL models
dl_models=(
    "AhnCNN AhnCNNModel TrainerDL"
    "AhnRNN AhnRNNModel TrainerDL"
    "BEyeLSTMArgs BEyeLSTMModel TrainerDL"
    "PLMASArgs PLMASModel TrainerDL"
    "MAG MAGModel TrainerDL"
    "PLMASfArgs PLMASFModel TrainerDL"
    "PostFusion PostFusionModel TrainerDL"
    "Roberta Roberteye TrainerDL"
    "RoberteyeFixation Roberteye TrainerDL"
    "RoberteyeWord Roberteye TrainerDL"
)

ml_models_cls=(
    "SupportVectorMachineMLArgs SupportVectorMachineMLModel TrainerML"
    # "XGBoostMLArgs XGBoostMLModel TrainerML"
    "RandomForestMLArgs RandomForestMLModel TrainerML"
    "LogisticRegressionMLArgs LogisticRegressionMLModel TrainerML"
    "LogisticMeziereArgs LogisticRegressionMLModel TrainerML"
    "DummyClassifierMLArgs DummyClassifierMLModel TrainerML"
)

ml_models_reg=(
    "SupportVectorRegressorMLArgs SupportVectorMachineMLModel TrainerML"
    # "XGBoostRegressorMLArgs XGBoostMLModel TrainerML"
    "RandomForestRegressorMLArgs RandomForestMLModel TrainerML"
    "LinearRegressionArgs LogisticRegressionMLModel TrainerML"
    "LinearMeziereArgs LogisticRegressionMLModel TrainerML"
    "DummyRegressorMLArgs DummyClassifierMLModel TrainerML"
)

if [[ "$regression" == "true" ]]; then
    model_base_trainer=("${dl_models[@]}" "${ml_models_reg[@]}")
else
    model_base_trainer=("${dl_models[@]}" "${ml_models_cls[@]}")
fi

if [[ "$train" == "true" ]]; then
    for fold in "${folds[@]}"; do
        for tuple in "${model_base_trainer[@]}"; do
            IFS=' ' read -r model base_model trainer <<<"$tuple"
            for data_task in "${data_tasks[@]}"; do
                echo "Running TRAIN model: $model ($base_model), data_task: $data_task, fold: $fold"
                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/run/single_run/train.py +model="$model" +data="$data_task" data.fold_index="$fold" +trainer="$trainer" trainer.overwrite_data="$overwrite_data" trainer.wandb_job_type=DEBUG trainer.run_mode=DEBUG
                if [ $? -ne 0 ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Data Task: $data_task, Trainer: $trainer, Fold: $fold, stage: Train, Model: $model" >>logs/failed_runs.log
                else
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Data Task: $data_task, Trainer: $trainer, Fold: $fold, stage: Train, Model: $model, " >>logs/completed_runs.log
                fi
            done
        done
    done
fi
if [[ "$test" == "true" ]]; then
    for tuple in "${dl_models[@]}"; do
        IFS=' ' read -r model base_model trainer <<<"$tuple"
        for data_task in "${data_tasks[@]}"; do
                echo "Running TEST model: $model ($base_model), data_task: $data_task"
                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/run/single_run/test_dl.py eval_path=\'outputs/+data=$data_task,+model=$model,+trainer=TrainerDL,trainer.overwrite_data=True,trainer.run_mode=DEBUG,trainer.wandb_job_type=DEBUG\'
                if [ $? -ne 0 ]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Data Task: $data_task, Trainer: $trainer, stage: Test, Model: $model" >>logs/failed_runs.log
                else
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Data Task: $data_task, Trainer: $trainer, stage: Test, Model: $model" >>logs/completed_runs.log
                fi
        done
    done
fi
