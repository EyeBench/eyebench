"""
This script creates and launches wandb sweeps for different models, data tasks, and trainers.
It generates bash & slurm scripts for running the sweeps on multiple GPUs or in a slurm environment.
It uses the wandb library to create and manage the sweeps.
"""

import os
import stat
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from tap import Tap

import wandb
from src.run.multi_run.search_spaces import search_space_by_model
from src.run.multi_run.utils import count_hyperparameter_configs

logger.add('logs/sweeps.log', level='INFO')


class HyperArgs(Tap):
    """
        Usage:
        1. check that 'search_space_by_model' has the correct hyperparameter search space
    for the model you wish to sweep.
        2. run 'python src/run/multi_run/sweep_creator.py --models <models>
    --data_task_names <data_task_names> --trainer_names <trainer_names>'
           * To run multiple models/data_tasks/trainers, separate them with spaces
        3. the script will create executable bash scripts for each sweep (fold_idx),
    which will launch the wandb sweeps.
        4. run the bash script ./<bash_script>.sh
        5. If you want to run on multiple GPUs, use the --gpu_count flag. Not tested for >1 GPUs.
    """

    run_cap: int = (
        250  # Maximum number of runs to execute. Relevant for non-grid search.
    )
    wandb_project: str = 'debug'  # Name of the wandb project to log to.
    wandb_entity: str = 'EyeRead'  # Name of the wandb entity to log to.
    folds: list[int] = [0]  # List of fold indices to run.
    gpu_count: int = 1  # Number of GPUs to use. >1  not tested.
    search_algorithm: Literal['bayes', 'grid', 'random'] = (
        'grid'  # Search algorithm to use.
    )

    # Slurm settings.
    slurm_cpus: int = 10  # Number of CPUs to use. Ideally number of workers + 2.
    slurm_mem: str = '75G'  # Amount of memory to use.
    slurm_mailto: str = 'shubi@campus.technion.ac.il'  # Email to send notifications to.
    num_duplicates_per_gpu: int = 1  # Number of duplicates to run on each GPU.

    # Model, data, and trainer settings as lists to support multiple values
    models: list[str] = []  # List of models to sweep
    base_models: list[str] = []  # List of base models to sweep
    data_tasks: list[str] = []  # List of data tasks to sweep
    trainers: list[str] = [
        'TrainerDL'
    ]  # List of trainers to sweep (default is 'default')

    # Filled in by the script
    trainer: str | None = None
    data_task: str | None = None
    model: str | None = None
    base_model: str | None = None


def create_sweep_configs(args: HyperArgs) -> list[dict]:
    """
    Create sweep configurations for the given hyperparameters.

    Args:
        args (HyperArgs): Hyperparameters for the sweep.

    Returns:
        list[dict]: List of sweep configurations.
    """
    search_space = search_space_by_model[args.base_model]
    logger.info(f'Creating sweep configs for {args.base_model}')
    _, total_count = count_hyperparameter_configs(search_space)
    logger.info(args)
    if total_count > args.run_cap:
        logger.warning(
            f'Warning: The number of hyperparameter configurations ({total_count}) is less than the run cap ({args.run_cap}).'
        )

    sweep_configs = [
        {
            'program': 'src/run/single_run/train.py',
            'method': args.search_algorithm,
            'metric': {
                'goal': 'minimize',
                'name': 'loss/val_all',
            },
            'entity': args.wandb_entity,
            'project': args.wandb_project,
            'name': f'{args.model}_{args.data_task}_fold_{fold_idx}',
            'parameters': search_space,
            'run_cap': args.run_cap,
            'command': [
                '${env}',
                '${interpreter}',
                '${program}',
                '${args_no_hypens}',
                f'+model={args.model}',
                f'+data={args.data_task}',
                f'+trainer={args.trainer}',
                f'data.fold_index={fold_idx}',
                f'trainer.devices={args.gpu_count}',
                f'trainer.wandb_job_type={args.model}_{args.data_task}',
            ],
        }
        for fold_idx in args.folds
    ]

    return sweep_configs


def launch_sweeps(entity: str, project: str, sweep_configs: list[dict]) -> list[str]:
    """
    Launch wandb sweeps for the given configurations.

    Args:
        entity (str): Name of the wandb entity.
        project (str): Name of the wandb project.
        sweep_configs (List[Dict]): List of sweep configurations.

    Returns:
        List[str]: List of sweep ids.
    """
    sweep_ids = [
        wandb.sweep(cfg, entity=entity, project=project) for cfg in sweep_configs
    ]
    return sweep_ids


def write_bash_script(
    filename: Path,
    main_command: str,
    sweep_ids: list[str],
    mode: Literal['lacc', 'david'],
) -> None:
    """
    Write a bash script to launch wandb agents in tmux sessions.

    Args:
        filename (str): Name of the bash script file.
        main_command (str): Main command to run in the tmux session.
        sweep_ids (list[str]): List of sweep ids.
    """
    if mode == 'lacc':
        conda_path = 'source $HOME/miniforge3/etc/profile.d/conda.sh'
        cd_path = 'cd $HOME/eyebench_private'
    elif mode == 'david':
        conda_path = 'source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh'
        cd_path = 'cd /mnt/mlshare/reich3/eyebench_private'
    else:
        raise ValueError(f'Invalid mode: {mode}')

    full_command = f"""#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

{conda_path}
{cd_path}

GPU_NUM=$1
RUNS_ON_GPU=${{2:-1}}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${{GPU_NUM}}-dup${{i}}-unified-{sweep_ids[0]}-{len(sweep_ids)}"
    tmux new-session -d -s "${{session_name}}" "{main_command}"; tmux set-option -t "${{session_name}}" remain-on-exit off
    echo "Launched W&B agent for GPU ${{GPU_NUM}}, Dup ${{i}} in tmux session ${{session_name}}"
done
"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_command)
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
    logger.info(f'Created bash script: {filename}')


def write_slurm_script(
    filename: Path,
    hyper_args: HyperArgs,
    sweep_ids: list[str],
    slurm_qos: Literal['normal', 'basic'],
) -> None:
    """
    Write a slurm script to launch wandb agents in slurm jobs.

    Args:
        filename (str): Name of the slurm script file.
        hyper_args (HyperArgs): Hyperparameters for the sweep.
        sweep_ids (list[str]): List of sweep ids.
        slurm_qos (str): Slurm quality of service (normal or basic).
    """
    base_srun_command = f"""
srun --overlap --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK -p work,mig \\
    --container-image=/rg/berzak_prj/shubi/prj/rev05_pytorchlightning+pytorch_lightning.sqsh \\
    --container-mounts="/rg/berzak_prj/shubi:/home/shubi" \\
    --container-workdir=/home/shubi/eyebench_private \\
    bash -c "
echo 'Starting job on $(date)'
source /home/shubi/prj/nvidia_pytorch_25_03_py3_mamba_wrapper.sh
conda activate eyebench
wandb agent {hyper_args.wandb_entity}/{hyper_args.wandb_project}/$SWEEP_ID"
    """
    srun_command = f"""{base_srun_command}"""
    if hyper_args.num_duplicates_per_gpu > 1:
        srun_command = f"""{srun_command}\nsleep 600\nwait"""
        for _ in range(hyper_args.num_duplicates_per_gpu - 1):
            srun_command += f"""{base_srun_command}\nsleep 10\n"""
        srun_command += 'wait'

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(
            f"""#!/bin/bash

#SBATCH --job-name={hyper_args.model}_{hyper_args.data_task}-array
#SBATCH --output=logs/{hyper_args.model}_{hyper_args.data_task}-%A_%a.out
#SBATCH --error=logs/{hyper_args.model}_{hyper_args.data_task}-%A_%a.err
#SBATCH --partition=work,mig
#SBATCH --ntasks={hyper_args.num_duplicates_per_gpu}
#SBATCH --nodes=1
#SBATCH --gpus={hyper_args.gpu_count}
#SBATCH --qos={slurm_qos}
#SBATCH --cpus-per-task={hyper_args.slurm_cpus}
#SBATCH --mem={hyper_args.slurm_mem}
#SBATCH --array=0-{len(sweep_ids) - 1}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={hyper_args.slurm_mailto}

sweep_ids=({' '.join(sweep_ids)})
SWEEP_ID=${{sweep_ids[$SLURM_ARRAY_TASK_ID]}}

{srun_command}
"""
        )
    os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
    logger.info(f'Created Slurm array job script: {filename}')


def create_bash_scripts(
    hyper_args: HyperArgs, sweep_ids: list[str], mode: Literal['lacc', 'david']
) -> None:
    """
    Create bash scripts for the given sweep ids.
    Args:
        hyper_args (HyperArgs): Hyperparameters for the sweep.
        sweep_ids (list[str]): List of sweep ids.
    """
    assert hyper_args.model is not None
    base_path = (
        Path('sweeps') / hyper_args.wandb_project / 'bash' / mode / hyper_args.model
    )
    base_path.mkdir(parents=True, exist_ok=True)
    filename = base_path / (
        f'{hyper_args.model}_{hyper_args.data_task}_folds_'
        + '_'.join(map(str, hyper_args.folds))
        + '.sh'
    )
    main_command = '; '.join(
        ['conda activate eyebench']
        + [
            f'CUDA_VISIBLE_DEVICES=${{GPU_NUM}} wandb agent '
            f'{hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep_id}'
            for sweep_id in sweep_ids
        ]
    )
    write_bash_script(
        filename=filename, main_command=main_command, sweep_ids=sweep_ids, mode=mode
    )


def create_slurm_scripts(
    hyper_args: HyperArgs, sweep_ids: list[str], slurm_qos: Literal['normal', 'basic']
) -> None:
    """
    Create slurm scripts for the given sweep ids.
    Args:
        hyper_args (HyperArgs): Hyperparameters for the sweep.
        sweep_ids (list[str]): List of sweep ids.
        slurm_qos (str): 'normal' or 'basic' for DGX.
    """
    assert hyper_args.model is not None
    base_path = Path('sweeps') / hyper_args.wandb_project / 'slurm' / hyper_args.model
    base_path.mkdir(parents=True, exist_ok=True)
    filename = base_path / (
        f'{hyper_args.model}_{hyper_args.data_task}_folds_'
        + '_'.join(map(str, hyper_args.folds))
        + f'{slurm_qos}.job'
    )
    write_slurm_script(
        filename=filename,
        hyper_args=hyper_args,
        sweep_ids=sweep_ids,
        slurm_qos=slurm_qos,
    )


def main():
    hyper_args = HyperArgs().parse_args()

    # Check if required parameters are provided
    if not hyper_args.models:
        logger.error('No models specified. Use --models to specify one or more models')
        return

    if not hyper_args.data_tasks:
        logger.error(
            'No data tasks specified. Use --data_tasks to specify one or more data tasks'
        )
        return

    logger.info(f'Hyper Args:\n{hyper_args}')
    logger.info(f'Models: {hyper_args.models}')
    logger.info(f'Data Tasks: {hyper_args.data_tasks}')
    logger.info(f'Trainers: {hyper_args.trainers}')

    # Loop over all combinations of models, data_tasks, and trainers
    all_sweep_ids = []

    for base_model, model, trainer in zip(
        hyper_args.base_models, hyper_args.models, hyper_args.trainers
    ):
        for data_task in hyper_args.data_tasks:
            # Set the current values for this iteration
            hyper_args.model = model
            hyper_args.data_task = data_task
            hyper_args.trainer = trainer
            hyper_args.base_model = base_model

            logger.info(f'\nCreating sweep for: {model=}, {data_task=}, {trainer=}')

            # Create and launch sweeps for this combination
            sweep_configs = create_sweep_configs(hyper_args)
            sweep_ids = launch_sweeps(
                entity=hyper_args.wandb_entity,
                project=hyper_args.wandb_project,
                sweep_configs=sweep_configs,
            )
            all_sweep_ids.extend(sweep_ids)

            create_slurm_scripts(hyper_args, sweep_ids, slurm_qos='normal')
            create_slurm_scripts(hyper_args, sweep_ids, slurm_qos='basic')
            create_bash_scripts(hyper_args, sweep_ids, mode='lacc')
            create_bash_scripts(hyper_args, sweep_ids, mode='david')

            # Save sweep ids to a file together with the hyper_args for this combination
            base_path = Path('sweeps') / hyper_args.wandb_project / 'configs'
            base_path.mkdir(parents=True, exist_ok=True)
            filename = base_path / f'{model}_{data_task}.yaml'
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(
                    {'hyper_args': hyper_args.as_dict(), 'sweep_ids': sweep_ids},
                    f,
                    default_flow_style=False,
                )

    logger.info(f'\nCreated a total of {len(all_sweep_ids)} sweeps')


if __name__ == '__main__':
    main()
