"""Main file for testing cognitive state decoding models"""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import lightning_fabric as lf
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from loguru import logger

from src.configs.constants import REGIMES
from src.configs.main_config import Args, ModelFactory
from src.configs.models.base_model import DLModelArgs
from src.configs.trainers import TrainerDL
from src.data.datamodules.base_datamodule import DataModuleFactory
from src.run.multi_run import supported_datamodules, supported_models  # noqa: F401
from src.run.single_run.utils import (
    configure_trainer,
    extract_trial_info,
    get_checkpoint_path,
    get_config,
)

cs = ConfigStore.instance()
cs.store(name='config', node=Args)


def get_fold_paths(base_path: Path) -> list[Path]:
    """Return sorted fold directories under the provided evaluation path."""
    fold_paths: list[Path] = []
    for path in base_path.glob('fold_index=*'):
        if not path.is_dir():
            continue
        try:
            int(path.name.split('=')[1])
        except (IndexError, ValueError):
            logger.warning(f'Skipping unexpected directory {path.name}')
            continue
        fold_paths.append(path)
    fold_paths.sort(key=lambda p: int(p.name.split('=')[1]))
    return fold_paths


@hydra.main(version_base=None, config_name='config')
def main(
    cfg: Args,
) -> None:
    lf.seed_everything(42, workers=True, verbose=False)
    torch.set_float32_matmul_precision('high')
    assert cfg.eval_path is not None, 'eval_path must be specified!'
    base_path = Path(cfg.eval_path)

    checkpoint_template = '*lowest_loss_val_all*.ckpt'

    found_at_least_one = False
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(logs_dir / f'{base_path}.log'), level='INFO')
    fold_paths = get_fold_paths(base_path)

    if not fold_paths:
        logger.error(f'No fold directories found under {base_path}!')
        exit(1)

    for fold_path in fold_paths:
        fold_index = int(fold_path.name.split('=')[1])
        try:
            cfg = get_config(config_path=fold_path)
            checkpoint_path = get_checkpoint_path(
                fold_path,
                checkpoint_template,
            )
            logger.info(f'Loading checkpoint from {checkpoint_path}')
            assert isinstance(cfg.model, DLModelArgs)
            model_class = ModelFactory.get(cfg.model.base_model_name)

            # Hack to load the model with the correct model_args
            model = model_class.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
            )
            # Recreate the config with the correct args (we still need cfg for the model class!)
            data_args = model.hparams['data_args']
            trainer_args = model.hparams['trainer_args']
            model_args = model.hparams['model_args']
            model_args.is_training = False
            cfg = Args(
                data=data_args,
                model=model_args,
                trainer=trainer_args,
            )

            # Note, important to load with the correct model_args
            model = model_class.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_args=model_args,
            )
            model.eval()

            assert isinstance(cfg.trainer, TrainerDL)
            trainer = configure_trainer(args=cfg.trainer, is_training=False)
        except AssertionError as e:
            logger.warning(f'Skipping fold {fold_index}! {e}')
            continue
        except FileNotFoundError as e:
            logger.warning(f'Skipping fold {fold_index}! {e}')
            continue

        dm = DataModuleFactory.get(datamodule_name=cfg.data.datamodule_name)(cfg)
        # dm.cfg.trainer.overwrite_data = True
        results = trainer.predict(model, datamodule=dm)

        assert results is not None, 'Results are None!'
        group_level_metrics = []
        for index, eval_type_results in enumerate(results):
            # based on predict_dataloader (first 3 are val, last three test)
            eval_type = 'val' if index in [0, 1, 2] else 'test'
            if eval_type == 'val':
                dataset = dm.val_datasets[index]
            else:
                dataset = dm.test_datasets[index % 3]

            labels, preds = zip(*eval_type_results)

            df = pd.DataFrame(
                {
                    'label': torch.cat(labels, dim=0).numpy(),
                    'prediction_prob': torch.cat(
                        [
                            p.unsqueeze(0)
                            if p.dim() == 0
                            else (
                                p.squeeze(0) if (p.dim() >= 1 and p.size(0) > 1) else p
                            )
                            for p in preds
                        ],
                        dim=0,
                    )
                    .numpy()
                    .tolist(),
                    'eval_regime': REGIMES[index % 3],
                    'eval_type': eval_type,
                    'fold_index': fold_index,
                },
            )

            trial_info = extract_trial_info(
                dataset, cols_to_keep=cfg.data.groupby_columns
            ).reset_index(drop=True)
            group_level_metrics.append(pd.concat([df, trial_info], axis=1))
            temp_path = fold_path / 'trial_level_test_results.csv'
            pd.concat(group_level_metrics).to_csv(
                temp_path,
                index=False,
            )
            found_at_least_one = True
            logger.info(f'Saving fold index {fold_index} to {temp_path}')

    if not found_at_least_one:
        logger.error('No results found!')
        exit(1)

    dst_root = Path('results/raw') / base_path.name
    dst_root.mkdir(parents=True, exist_ok=True)

    # copy all .hydra folders under this base_path
    for hydra_dir in base_path.rglob('.hydra'):
        rel_path = hydra_dir.relative_to(base_path)
        target = dst_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(hydra_dir, target, dirs_exist_ok=True)
        logger.info(f'Copied {hydra_dir} → {target}/')

    # copy fold level results
    for fold_path in fold_paths:
        fold_index = int(fold_path.name.split('=')[1])
        fold_results_path = fold_path / 'trial_level_test_results.csv'
        if fold_results_path.exists():
            shutil.copy(
                fold_results_path, dst_root / f'{fold_index=}' / fold_results_path.name
            )
            logger.info(f'Copied {fold_results_path} → {dst_root}/')
    # ------------------------------------------------------------------------


if __name__ == '__main__':
    main()
