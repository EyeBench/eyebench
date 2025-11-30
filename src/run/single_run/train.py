"""Main file for cognitive state decoding training"""

from __future__ import annotations

import dataclasses
import os
import pprint
from typing import Sequence, cast

import hydra
import lightning_fabric as lf
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig

import wandb
from src.configs.constants import DLModelNames, RunModes
from src.configs.main_config import Args, get_model
from src.configs.models.base_model import DLModelArgs
from src.configs.models.dl.BEyeLSTM import BEyeLSTMArgs
from src.configs.trainers import TrainerDL
from src.data.datamodules.base_datamodule import DataModuleFactory, ETDataModule
from src.data.datasets.base_dataset import ETDataset
from src.models.base_model import BaseModel
from src.run.multi_run import supported_datamodules, supported_models  # noqa: F401
from src.run.single_run.utils import (
    configure_trainer,
    instantiate_config,
    setup_logger,
    update_cfg_with_wandb,
)

cs = ConfigStore.instance()
cs.store(name='config', node=Args)


def _configure_beye_lstm_model(args: Args, train_dataset: ETDataset) -> Args:
    if not isinstance(args.model, BEyeLSTMArgs):
        return args

    feature_names = cast(Sequence[object], train_dataset.trial_level_feature_names)
    args.model.gsf_dim = len(feature_names)
    logger.info(f'GSF dim: {args.model.gsf_dim} (number of trial-level features)')

    return args


def _configure_roberteye_model(args: Args, train_dataset: ETDataset) -> Args:
    if args.model.base_model_name not in [DLModelNames.ROBERTEYE_MODEL]:
        return args

    args.model.n_tokens = train_dataset.n_tokens
    args.model.sep_token_id = train_dataset.sep_token_id
    args.model.eye_token_id = train_dataset.eye_token_id
    args.model.is_training = True

    return args


def _configure_postfusion_model(args: Args, train_dataset: ETDataset) -> Args:
    if args.model.base_model_name not in [DLModelNames.POSTFUSION_MODEL]:
        return args

    args.model.sep_token_id = train_dataset.sep_token_id

    return args


def _configure_dl_model(args: Args, dm: ETDataModule) -> Args:
    train_dataset = dm.train_dataset

    args = _configure_beye_lstm_model(args=args, train_dataset=train_dataset)
    args = _configure_roberteye_model(args=args, train_dataset=train_dataset)
    args = _configure_postfusion_model(args=args, train_dataset=train_dataset)

    return args


@hydra.main(version_base=None, config_name='config')
def main(cfg: DictConfig) -> None:
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        logger.error('Note: CUDA_VISIBLE_DEVICES is not set!')

    args = instantiate_config(cfg=cfg)
    lf.seed_everything(seed=args.trainer.seed, workers=True, verbose=False)
    torch.set_float32_matmul_precision(precision=args.trainer.float32_matmul_precision)

    work_dir = HydraConfig.get().runtime.output_dir

    if args.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.init(
            entity=args.trainer.wandb_entity,
            project=args.trainer.wandb_project,
            job_type=args.trainer.wandb_job_type,
            notes=args.trainer.wandb_notes,
            dir=work_dir,
        )

    # # If wandb config is not empty, we are running a sweep, so we need to update the args.
    if wandb.run and wandb.config.as_dict():
        args = update_cfg_with_wandb(args)

    # log the config to wandb
    if args.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.config.update(dataclasses.asdict(args))

    pprint.pprint(args)

    dm = DataModuleFactory.get(datamodule_name=args.data.datamodule_name)(args)
    dm.prepare_data()
    dm.setup(stage='fit')

    # Update class weights only if weighting
    if args.model.use_class_weighted_loss:
        class_weights = ETDataset.organize_label_counts(
            dm.train_dataset.labels.tolist(), label_names=args.data.class_names
        )['count']
        args.model.class_weights = (sum(class_weights) / class_weights).tolist()
        logger.info(f'Class weights: {args.model.class_weights}')

    if not args.model.is_ml:
        args = _configure_dl_model(args=args, dm=dm)

    model = get_model(args)

    if args.model.is_ml:
        model.fit(dm=dm)

        # evaluate the model on train
        model.evaluate(
            eval_dataset=dm.train_dataset, stage='train', validation_map='all'
        )
        model.on_stage_end()

        # evaluate the model on val
        for validation_map, val_dataset in zip(
            model.regime_names,
            dm.val_datasets,
        ):  # drops the 'all' validation map as datasets don't include it
            model.evaluate(
                eval_dataset=val_dataset,
                stage='val',
                validation_map=validation_map,
            )
        model.on_stage_end()
    else:
        assert isinstance(args.trainer, TrainerDL)
        assert isinstance(args.model, DLModelArgs)
        assert isinstance(model, BaseModel)
        if args.trainer.run_mode != RunModes.FAST_DEV_RUN:
            wandb_logger = setup_logger(
                wandb_entity=args.trainer.wandb_entity,
                wandb_project=args.trainer.wandb_project,
                wandb_job_type=args.trainer.wandb_job_type,
            )
            if args.trainer.log_gradients:
                wandb_logger.watch(model=model)
        else:
            wandb_logger = None

        trainer = configure_trainer(
            args=args.trainer,
            wandb_logger=wandb_logger,
            work_dir=work_dir,
            accumulate_grad_batches=args.model.accumulate_grad_batches,
            max_time=args.model.max_time,
            max_epochs=args.model.max_epochs,
            early_stopping_patience=args.model.early_stopping_patience,
            is_training=True,
        )
        trainer.fit(model=model, datamodule=dm)

        checkpoint_callback = getattr(trainer, 'checkpoint_callback', None)
        if isinstance(checkpoint_callback, ModelCheckpoint):
            logger.info(checkpoint_callback.best_model_path)

    if args.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.finish()


if __name__ == '__main__':
    main()
