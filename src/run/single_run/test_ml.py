"""
__summary__:
    Given a list of experiments, where each experiment is a dictionary
    that maps fold_idx to a completed w&b sweep, this script ought to
    1. take the best hyperparameters from each fold (sweep) or from a
       designated run_id if requested,
    2. fit the model on that fold,
    3. save test predictions to file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightning_fabric as lf
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from loguru import logger
from tap import Tap
from tqdm import tqdm

from src.configs.constants import REGIMES, MLModelNames, Scaler
from src.configs.data import DATA_CONFIGS_MAPPING
from src.configs.main_config import Args, get_model
from src.configs.models.base_model import MLModelArgs
from src.configs.models.ml.DummyClassifier import (
    DummyClassifierMLArgs,
    DummyRegressorMLArgs,
)  # noqa: F401
from src.configs.models.ml.LogisticRegression import (
    LinearMeziereArgs,
    LinearRegressionArgs,
    LogisticMeziereArgs,
    LogisticRegressionMLArgs,
)  # noqa: F401
from src.configs.models.ml.RandomForest import (
    RandomForestMLArgs,
    RandomForestRegressorMLArgs,
)  # noqa: F401
from src.configs.models.ml.SVM import (
    SupportVectorMachineMLArgs,
    SupportVectorRegressorMLArgs,
)  # noqa: F401
from src.configs.models.ml.XGBoost import (  # noqa: F401
    XGBoostMLArgs,
    XGBoostRegressorMLArgs,
)
from src.configs.trainers import TrainerML
from src.data.datamodules import base_datamodule
from src.data.datamodules.base_datamodule import DataModuleFactory
from src.run.multi_run import supported_datamodules, supported_models  # noqa: F401
from src.run.single_run.utils import extract_trial_info

CLASSIFICATION_MODEL_CONFIGS = [
    SupportVectorMachineMLArgs,
    # XGBoostMLArgs,
    RandomForestMLArgs,
    LogisticRegressionMLArgs,
    LogisticMeziereArgs,
    DummyClassifierMLArgs,
]  # noqa: F401
REGRESSION_MODEL_CONFIGS = [
    SupportVectorRegressorMLArgs,
    # XGBoostRegressorMLArgs,
    RandomForestRegressorMLArgs,
    LinearRegressionArgs,
    LinearMeziereArgs,
    DummyRegressorMLArgs,
]  # noqa: F401


def main() -> None:
    args = HyperArgs().parse_args()

    # Get the data config and determine if it's regression using the is_regression property
    data_config = DATA_CONFIGS_MAPPING[args.data_task]
    data_instance = data_config()

    model_configs = (
        REGRESSION_MODEL_CONFIGS
        if data_instance.is_regression
        else CLASSIFICATION_MODEL_CONFIGS
    )

    experiments = [
        Experiment(
            model_args=model,
            data_args=data_config,
            wandb_project=args.wandb_project,
        )
        for model in model_configs
    ]  # type: ignore
    checks(experiments)

    lf.seed_everything(42, workers=True, verbose=False)
    torch.set_float32_matmul_precision('high')
    api = wandb.Api()

    # Single run if run_id is provided
    if args.wandb_run_id is not None:
        cfg_of_run = get_config_from_run(
            api,
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_id=args.wandb_run_id,
        )
        trainer_args = TrainerML(**cfg_of_run['trainer'])
        data_args = DATA_CONFIGS_MAPPING['OneStop_RC'](
            **cfg_of_run['data']
        )  # TODO use datafactory to generalize

        model_args = XGBoostMLArgs(
            **cfg_of_run['model']
        )  # TODO make use of modelfactory

        process_single_run(
            data_args=data_args,
            trainer_args=trainer_args,
            model_args=model_args,
            fold_index=data_args.fold_index,
        )

    else:  # Process all experiments and sweeps
        checks(experiments)
        # For each experiment
        for exp in experiments:
            for sweep in tqdm(exp.sweeps):
                cfg_of_best = get_config_from_sweep(
                    api,
                    args.wandb_entity,
                    exp.wandb_project,
                    sweep.sweep_id,
                )
                sweep.cfg_of_best = cfg_of_best
                sweep.fold_index = int(cfg_of_best['data']['fold_index'])

                trainer_args = TrainerML(**sweep.cfg_of_best['trainer'])
                data_args = exp.data_args(**sweep.cfg_of_best['data'])
                model_args = exp.model_args(**sweep.cfg_of_best['model'])

                process_single_run(
                    data_args, trainer_args, model_args, fold_index=sweep.fold_index
                )

            logger.info(
                f'{exp.wandb_project} - {exp.model_name} - {exp.data_args} done'
            )


class HyperArgs(Tap):
    """
    Command line arguments for the script.
    """

    wandb_entity: str = 'EyeRead'  # Name of the wandb entity to log to.
    wandb_run_id: str | None = None  # Provide if you want a single run.
    data_task: str = 'CopCo_TYP'  # Name of the data task (e.g., CopCo_TYP).
    wandb_project: str = 'CopCo_TYP_20250714'  # Name of the wandb project.


@dataclass
class Sweep:
    """
    Class representing a sweep in wandb.

    Attributes:
        sweep_id (str): The ID of the sweep in wandb.
        cfg_of_best (dict): The configuration of the best run in the sweep.
        fold_index (int | None): The index of the fold, if applicable.
    """

    sweep_id: str
    cfg_of_best: dict = field(default_factory=dict)
    fold_index: int | None = None


@dataclass
class Experiment:
    """
    Class representing an experiment.
    """

    dataset_name: str = field(init=False)
    model_name: str = field(init=False)
    model_args: dataclass
    data_args: dataclass
    save_folder_name: Path = field(default_factory=Path)
    sweeps: list[Sweep] = field(default_factory=list)
    wandb_project: str = 'ml_debug'

    def load_sweep_ids_from_yaml(self, yaml_path: str) -> list[str]:
        """
        Load sweep IDs from a YAML file.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            sweep_cfg = yaml.safe_load(f)
        return sweep_cfg.get('sweep_ids', [])

    def __post_init__(self):
        self.dataset_name = self.data_args.__name__
        self.model_name = self.model_args.__name__
        # Load sweep IDs from the YAML file
        sweep_ids = self.load_sweep_ids_from_yaml(
            f'sweeps/{self.wandb_project}/configs/{self.model_name}_{self.dataset_name}.yaml'
        )
        # Create Sweep objects for each sweep ID
        self.sweeps = [Sweep(sweep_id=sweep_id) for sweep_id in sweep_ids]

        self.save_folder_name = Path(
            f'results/raw/+data={self.dataset_name},'
            f'+model={self.model_name},+trainer=TrainerML,trainer.wandb_job_type='
            f'{self.model_name}_{self.dataset_name}',
        )

        self.save_folder_name.mkdir(parents=True, exist_ok=True)


def get_config_from_sweep(
    api: wandb.Api,
    entity: str,
    project: str,
    sweep_id: str,
) -> dict[str, Any]:
    """
    Fetches the config of the *best run* (by the sweep's objective) from a given sweep_id.
    """
    sweep_obj = api.sweep(path=f'{entity}/{project}/{sweep_id}')
    best_run = sweep_obj.best_run()
    return best_run.config


def get_config_from_run(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> dict[str, Any]:
    """
    Fetches the config of a single run by run_id.
    """
    return api.run(path=f'{entity}/{project}/{run_id}').config


def checks(experiments_list: list[Experiment]) -> None:
    """
    Basic consistency check: ensure that each sweep_id is unique across experiments.
    """
    sweep_ids = [sweep.sweep_id for exp in experiments_list for sweep in exp.sweeps]
    assert len(sweep_ids) == len(set(sweep_ids)), 'Duplicate sweep IDs found!'


def predict_on_val_and_test(
    model: torch.nn.Module,
    val_datasets: list[np.ndarray],
    test_datasets: list[np.ndarray],
) -> list[np.ndarray]:
    """Predict on all val and test datasets, returning one list of results."""
    results = []
    # Predict on validation datasets
    for val_dataset in val_datasets:
        results.append(model.predict(val_dataset))

    # Predict on test datasets
    for test_dataset in test_datasets:
        results.append(model.predict(test_dataset))

    return results


def process_results(
    results: list[tuple[torch.Tensor, ...]],
    dm: base_datamodule.ETDataModuleFast,
    cfg: Args,
    fold_index: int,
) -> pd.DataFrame:
    """
    Given all results from val and test datasets, build
    a unified DataFrame with all relevant columns.
    # TODO almost duplicate code with test_dl.py
    """
    group_level_metrics = []

    for index, eval_type_results in enumerate(results):
        # based on predict_dataloader (first 3 are val, last three test)
        eval_type = 'val' if index in [0, 1, 2] else 'test'
        if eval_type == 'val':
            dataset = dm.val_datasets[index]
        else:
            dataset = dm.test_datasets[index % 3]

        # Decide whether we have grouped trial keys

        trial_info = extract_trial_info(
            dataset, cols_to_keep=cfg.data.groupby_columns
        ).reset_index(drop=True)

        # Unpack model outputs
        preds, probs, y_true = eval_type_results
        if probs is None:
            probs = preds
        df = pd.DataFrame(
            {
                'label': y_true.numpy(),
                'prediction_prob': probs.numpy().tolist(),
                'eval_regime': REGIMES[index % 3],
                'eval_type': eval_type,
                'fold_index': fold_index,
            },
        )

        group_level_metrics.append(pd.concat([df, trial_info], axis=1))

    res = pd.concat(group_level_metrics)

    return res


def process_single_run(data_args, trainer_args, model_args, fold_index) -> pd.DataFrame:
    # -------------------------------------
    # Single-run processing logic
    # -------------------------------------

    args = Args(
        data=data_args,
        model=model_args,
        trainer=trainer_args,
    )

    # TODO copied from src.run.multi_run.utils.instatiate_config
    args.data.full_dataset_name = args.data.__class__.__name__
    args.model.full_model_name = args.model.model_name
    args.model.max_time_limit = args.model.max_time
    args.model.is_ml = args.model.base_model_name in MLModelNames
    args.model.use_class_weighted_loss = (
        args.model.use_class_weighted_loss
        if len(list(args.data.class_names)) > 1
        else False
    )

    args.model.normalization_type = Scaler.ROBUST_SCALER
    # args.trainer.overwrite_data = True
    # Replace 'UNIQUE_TRIAL_ID' with 'unique_trial_id' and 'SUBJECT_ID' with 'participant_id' in groupby_columns
    args.data.groupby_columns = [
        col.replace('UNIQUE_TRIAL_ID', 'unique_trial_id').replace(
            'SUBJECT_ID', 'participant_id'
        )
        for col in args.data.groupby_columns
    ]  # TODO Not pretty code
    dm = DataModuleFactory.get(datamodule_name=args.data.datamodule_name)(args)
    dm.prepare_data()
    dm.setup(stage='fit')

    assert isinstance(args.trainer, TrainerML)
    assert isinstance(args.model, MLModelArgs)
    model = get_model(cfg=args)

    model.fit(dm=dm)

    dm.setup(stage='predict')  # creates val and test sets

    # Use the new helper to gather predictions
    results = predict_on_val_and_test(
        model=model,
        val_datasets=dm.val_datasets,
        test_datasets=dm.test_datasets,
    )
    # Convert all predictions/results into one DataFrame
    res = process_results(
        results=results,
        dm=dm,
        cfg=args,
        fold_index=fold_index,
    )

    # Save results
    save_path = (
        Path('results/raw')
        / f'+data={args.data.full_dataset_name},+model={args.model.model_name},+trainer=TrainerML,trainer.wandb_job_type={args.model.model_name}_{args.data.full_dataset_name}'
        / f'{fold_index=}'
        / 'trial_level_test_results.csv'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    res.to_csv(save_path)
    logger.info(f'Single run results saved to {save_path}')
    return res


if __name__ == '__main__':
    main()
