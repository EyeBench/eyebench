#!/bin/bash
# sweep_wrapper.sh
# This script wraps the sweep_creator.py script, allowing you to specify data tasks, folds, and other options.
# Usage:
#   ./sweep_wrapper.sh [--data_tasks task1,task2,...] [--folds fold1,fold2,...] [--wandb_project name]
# Options:
#   --data_tasks   Comma-separated list of data tasks (default: CopCo_TYP)
#   --folds        Comma-separated list of folds (default: 0)
#   --wandb_project  Name of the wandb project (default: auto-generated)
#   --regression   Set to true for regression tasks (default: false)
#   -h, --help     Show this help message and exit
# The script will set the wandb_project variable to a default value based on the selected data tasks and folds,
# unless overridden.
# Example:
#   bash run_commands/utils/sweep_wrapper.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --regression true 

# Default values
data_tasks=("CopCo_RCS")
folds=(0 1 2 3)
wandb_project="CopCo_RCS"
run_cap=1000
regression="false"
print_help() {
    echo "Usage: $0 [--data_tasks task1,task2,...] [--folds fold1,fold2,...] [--wandb_project name]"
    echo "Options:"
    echo "  --data_tasks     Comma-separated list of data tasks (default: CopCo_TYP)"
    echo "  --folds          Comma-separated list of folds (default: 0)"
    echo "  --wandb_project  Name of the wandb project (default: auto-generated)"
    echo "  -h, --help       Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --data_tasks)
        IFS=',' read -ra data_tasks <<<"$2"
        shift 2
        ;;
    --folds)
        IFS=',' read -ra folds <<<"$2"
        shift 2
        ;;
    --wandb_project)
        wandb_project="$2"
        shift 2
        ;;
    --regression)
        regression="$2"
        shift 2
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
    esac
done

# Set default wandb_project if not provided
if [[ -z "$wandb_project" ]]; then
    wandb_project="${data_tasks[*]}_${folds[*]}"
    wandb_project="${wandb_project// /_}"
fi
echo "Wandb project: $wandb_project"
echo "Data tasks: ${data_tasks[@]}"
echo "Folds: ${folds[@]}"
echo "Regression: $regression"

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
    "XGBoostMLArgs XGBoostMLModel TrainerML"
    "RandomForestMLArgs RandomForestMLModel TrainerML"
    "LogisticRegressionMLArgs LogisticRegressionMLModel TrainerML"
    "LogisticMeziereArgs LogisticRegressionMLModel TrainerML"
    "DummyClassifierMLArgs DummyClassifierMLModel TrainerML"
)

ml_models_reg=(
    "SupportVectorRegressorMLArgs SupportVectorRegressorMLModel TrainerML"
    "XGBoostRegressorMLArgs XGBoostRegressorMLModel TrainerML"
    "RandomForestRegressorMLArgs RandomForestRegressorMLModel TrainerML"
    "LinearRegressionArgs LinearRegressionRegressorMLModel TrainerML"
    "LinearMeziereArgs LinearRegressionRegressorMLModel TrainerML"
    "DummyRegressorMLArgs DummyRegressorMLModel TrainerML"
)

if [[ "$regression" == "true" ]]; then
    model_base_trainer=("${dl_models[@]}" "${ml_models_reg[@]}")
else
    model_base_trainer=("${dl_models[@]}" "${ml_models_cls[@]}")
fi

models=()
base_models=()
trainers=()
for tuple in "${model_base_trainer[@]}"; do
    IFS=' ' read -r model base_model trainer <<<"$tuple"
    models+=("$model")
    base_models+=("$base_model")
    trainers+=("$trainer")
done
python src/run/multi_run/sweep_creator.py \
    --models "${models[@]}" \
    --base_models "${base_models[@]}" \
    --trainers "${trainers[@]}" \
    --data_tasks "${data_tasks[@]}" \
    --folds "${folds[@]}" \
    --wandb_project "${wandb_project}" \
    --run_cap "$run_cap"
