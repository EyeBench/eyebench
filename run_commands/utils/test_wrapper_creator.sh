#!/bin/bash
# test_wrapper_creator.sh
# This script generates a wrapper script for running test_dl.py on a single data task.
# Usage:
#   ./test_wrapper_creator.sh [--mode local|slurm] [--data_task name] [--project_name project1]
# Options:
#   --mode         "local" (default) or "slurm"
#   --data_task    Name of the data task (default: OneStop_RC)
#   --project_name  Output file for local mode (default: sweeps/<project_name>/test_dl_wrapper.sh)
#   -h, --help     Show this help message and exit
# Example:
#   bash run_commands/utils/test_wrapper_creator.sh --data_task CopCo_TYP --mode local --project_name CopCo_TYP

# Default values
mode="local"
data_task="SBSAT_STD"
project_name="SBSAT_STD__2305"
output_file=""

print_help() {
    echo "Usage: $0 [--mode local|slurm] [--data_task name] [--project_name project1]"
    echo "Options:"
    echo "  --mode         \"local\" (default) or \"slurm\""
    echo "  --data_task    Name of the data task (default: OneStop_RC)"
    echo "  --project_name  Output file for local mode (default: sweeps/<project_name>/test_dl_wrapper.sh)"
    echo "  -h, --help     Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --data_task)
        data_task="$2"
        shift 2
        ;;
    --project_name)
        project_name="$2"
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

if [[ -z "$output_file" ]]; then
    output_file="sweeps/${project_name}/test_dl_wrapper.sh"
    echo output_file
fi


# common config
base_path=outputs

model_base_trainer=(
    "AhnCNN AhnCNN TrainerDL"
    "AhnRNN AhnRNN TrainerDL"
    "BEyeLSTMArgs BEyeLSTMArgs TrainerDL"
    "MAG MAG TrainerDL"
    "PLMASArgs PLMASArgs TrainerDL"
    "PLMASfArgs PLMASfArgs TrainerDL"
    "PostFusion PostFusion TrainerDL"
    "Roberta Roberteye TrainerDL"
    "RoberteyeFixation Roberteye TrainerDL"
    "RoberteyeWord Roberteye TrainerDL"
)

# explode model_base_trainer tuples into separate arrays
models=()
base_models=()
trainers=()
for tuple in "${model_base_trainer[@]}"; do
    IFS=' ' read -r model base_model trainer <<<"$tuple"
    models+=("$model")
    base_models+=("$base_model")
    trainers+=("$trainer")
done

if [[ "$mode" == "slurm" ]]; then
    mkdir -p slurm_log
    for tuple in "${model_base_trainer[@]}"; do
        IFS=' ' read -r model base_model trainer <<<"$tuple"
        job_name="eval_${data_task}_${model}"
        sbatch_file="eval_${job_name}.job"
        cat >"$sbatch_file" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/slurm-${job_name}-%j.out
#SBATCH --error=logs/slurm-${job_name}-%j.err
#SBATCH --partition=work,mig
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

srun --ntasks=1 --nodes=1 --cpus-per-task=\$SLURM_CPUS_PER_TASK -p work,mig \\\n
    --container-image=/rg/berzak_prj/shubi/prj/rev05_pytorchlightning+pytorch_lightning.sqsh \\\n
    --container-mounts="/rg/berzak_prj/shubi:/home/shubi" \\\n
    --container-workdir=/home/shubi/eyebench \\\n
    bash -c "
echo 'Starting job on $(date)'
source /home/shubi/prj/nvidia_pytorch_25_03_py3_mamba_wrapper.sh
conda activate eyebench    
python src/run/single_run/test_dl.py "eval_path=\\"${base_path}/+data=${data_item},+model=${model},+trainer=${trainer},trainer.wandb_job_type=${model}_${data_item}\\""
EOF
            chmod +x "$sbatch_file"
    done
    echo "Generated Slurm job files for all model/data combinations."
else
    mkdir -p "$(dirname "$output_file")"
    {
        # printf "python src/run/multi_run/cleanup_models.py --real_run --keep_one_lowest\n\n"
        printf "paths=(\n"
        for tuple in "${model_base_trainer[@]}"; do
            IFS=' ' read -r model base_model trainer <<<"$tuple"
            printf "\"%s/+data=%s,+model=%s,+trainer=%s,trainer.wandb_job_type=%s_%s\"\n" \
                "$base_path" "$data_task" "$model" "$trainer" "$model" "$data_task"
        done
        printf ")\n\n"
        printf "# Use CUDA_VISIBLE_DEVICES from environment, default to 0 if not set\n"
        printf "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}\n\n"
        printf "for path in \"\${paths[@]}\"; do\n"
        printf "  export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} ; python src/run/single_run/test_dl.py \"eval_path=\\\\\"\${path}\\\\\"\"\n"
        printf "done\n"
    } >"$output_file"
    chmod +x "$output_file"
    echo "Generated local run script: $output_file"
fi
