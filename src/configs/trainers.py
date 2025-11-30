"""
This module contains dataclasses for configuring trainers.

The module defines a hierarchy of configuration classes:
- `BaseTrainer`: The root configuration class with common attributes
- `TrainerDL`: Configuration for deep learning trainers, inheriting from BaseTrainer
- `TrainerML`: Configuration for machine learning trainers, inheriting from BaseTrainer

Each configuration class is defined using the `@dataclass` decorator and specifies
the relevant attributes and their default values.

The `@register_config` decorator is used to register the configuration classes with a
specific group defined by `ConfigName.TRAINER`.
"""

from dataclasses import dataclass
from typing import Any

from src.configs.constants import (
    Accelerators,
    MatmulPrecisionLevel,
    Precision,
    RunModes,
)
from src.configs.utils import register_trainer


@dataclass
class BaseTrainer:
    """
    Base configuration class for all trainers.

    This class defines common attributes shared by both deep learning and machine learning trainers.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Default is 4.
        profiler (str | None): Profiler to use ('simple', 'advanced', or None). Default is None.
        precision (Precision): Numerical precision for training.
            Default is Precision.THIRTY_TWO_TRUE.
        float32_matmul_precision (MatmulPrecisionLevel): Matrix multiplication precision level.
            Default is MatmulPrecisionLevel.HIGH.
        seed (int): Random seed for reproducibility. Default is 42.
        devices (Any): Device configuration for training. Default is 1.
        run_mode (RunModes): Mode for running the trainer (e.g., 'train', 'test', 'debug').
            Default is RunModes.TRAIN.
        wandb_job_type (str): Type of job for Weights & Biases logging. Default is "MISSING".
        wandb_project (str): Weights & Biases project name.
            Default is "reading-comprehension-from-eye-movements".
        wandb_entity (str): Weights & Biases entity name. Default is "EyeRead".
        wandb_notes (str): Additional notes for Weights & Biases logging.
            Default is an empty string.
        overwrite_data (bool): If True, overwrites the relevant TextDataSet and ETDataset.
            features even if they exist.
    """

    num_workers: int = 4
    profiler: str | None = None
    precision: Precision = Precision.THIRTY_TWO_TRUE
    float32_matmul_precision: MatmulPrecisionLevel = MatmulPrecisionLevel.HIGH
    seed: int = 42
    devices: Any = 1
    run_mode: RunModes = RunModes.TRAIN
    wandb_job_type: str = 'MISSING'
    wandb_project: str = 'EyeBench'
    wandb_entity: str = 'EyeRead'
    wandb_notes: str = ''
    sample_m_per_class: bool = False
    samples_per_epoch: int | None = None
    overwrite_data: bool = False

    def __post_init__(self):
        """
        Post-initialization hook to adjust attributes based on the run mode.

        If the run mode is set to 'debug', the number of workers is set to 0
        and the Weights & Biases job type is set to "debug".
        """
        if self.run_mode == RunModes.DEBUG:
            self.num_workers = 0
            self.wandb_job_type = 'debug'

        assert self.sample_m_per_class is False or (
            self.samples_per_epoch is not None
        ), 'samples_per_epoch must be set if sample_m_per_class is True'


@register_trainer
@dataclass
class TrainerDL(BaseTrainer):
    """
    Configuration class for deep learning trainers.

    Inherits from BaseTrainer and adds specific attributes for deep learning models.

    Attributes:
        learning_rate (float): Optimizer learning rate. Must be specified by derived classes.
        gradient_clip_val (float | None): Gradient clipping value. Default is None.
        accelerator (Accelerators): Accelerator to use (e.g., 'cpu', 'gpu', 'tpu').
            Default is Accelerators.AUTO.
        log_gradients (bool): Whether to log gradients. Default is False.
        optimize_for_loss (bool): Whether to optimize for loss instead of metrics. Default is True.
    """

    learning_rate: float = 0.0003
    gradient_clip_val: float | None = 1.0
    accelerator: Accelerators = Accelerators.AUTO
    log_gradients: bool = False
    optimize_for_loss: bool = True
    use_torch_compile: bool = False


@register_trainer
@dataclass
class TrainerML(BaseTrainer):
    """
    Configuration class for machine learning trainers.
    Inherits from BaseTrainer and adds specific attributes for machine learning models.
    """


@register_trainer
@dataclass
class SamplingTrainerDL(TrainerDL):
    sample_m_per_class: bool = True
    samples_per_epoch: int | None = (
        42_800 // 6
    )  # 42.8K corresponds to <1% not picking a specific sample given 4636 "A" samples.
