"""Configuration file for the model, trainer, data paths and data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from src.configs.data import DataArgs
from src.configs.models.base_model import (
    BaseModelArgs,
)
from src.configs.trainers import BaseTrainer
from src.models.base_model import BaseMLModel, BaseModel, ModelFactory


@dataclass
class Args:
    """
    Configuration class for the model, trainer, data paths, and data.

    Attributes:
        model (BaseModelArgs): Configuration for the model.
        trainer (BaseTrainer): Configuration for the trainer.
        data (DataArgs): Configuration for the data.
        eval_path (str | None): Path for evaluation.
        hydra (Any): Configuration for Hydra.
    """

    model: BaseModelArgs = field(default_factory=BaseModelArgs)
    trainer: BaseTrainer = field(default_factory=BaseTrainer)
    data: DataArgs = field(default_factory=DataArgs)
    eval_path: str | None = None
    # https://hydra.cc/docs/1.3/configure_hydra/workdir/
    hydra: Any = field(
        default_factory=lambda: {
            'run': {
                'dir': 'outputs/${hydra:job.override_dirname}/fold_index=${data.fold_index}',
            },
            'sweep': {
                'dir': 'cross_validation_runs',
                # https://github.com/facebookresearch/hydra/issues/1786#issuecomment-1017005470
                'subdir': '${hydra:job.override_dirname}/fold_index=${data.fold_index}',
            },
            'job': {
                'config': {
                    'override_dirname': {
                        # Don't include fold_index and devices in the directory name
                        'exclude_keys': [
                            'data.fold_index',
                            'trainer.devices',
                        ],
                    },
                },
            },
        },
    )


def get_model(cfg: Args) -> BaseModel | BaseMLModel:
    """
    Returns a model based on the model name.

    Args:
        cfg (Args): Configuration object containing model parameters.

    Returns:
        BaseModel: An instance of the model class.
    """

    model_class = ModelFactory.get(cfg.model.base_model_name)
    model = model_class(
        trainer_args=cfg.trainer,
        model_args=cfg.model,
        data_args=cfg.data,
    )

    if getattr(cfg.trainer, 'use_torch_compile', False):
        logger.info('Using torch.compile')
        model = torch.compile(
            model,
            mode='reduce-overhead',
        )

    return model
