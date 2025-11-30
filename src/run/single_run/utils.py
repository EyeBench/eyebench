from __future__ import annotations

import ast
import re
from collections import defaultdict
from datetime import timedelta
from os.path import join
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra import compose
from hydra.utils import instantiate, to_absolute_path
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import wandb
from src.configs.constants import (
    BackboneNames,
    DatasetLanguage,
    MLModelNames,
    Precision,
    RunModes,
    SetNames,
)
from src.configs.main_config import Args
from src.configs.models.base_model import DLModelArgs
from src.configs.models.dl.RoBERTeye import RoberteyeArgs
from src.configs.trainers import TrainerDL
from src.data.datasets.base_dataset import ETDataset


def instantiate_config(cfg: DictConfig) -> Args:
    """
    Instantiate the config object with the appropriate datamodule and model.

    Args:
        cfg (dict): The configuration object.

    Returns:
        Args: The instantiated configuration object.
    """
    args: Args = instantiate(config=cfg, _convert_='object')
    args.data.full_dataset_name = args.data.dataset_name
    args.model.full_model_name = args.model.model_name
    args.model.max_time_limit = args.model.max_time
    args.model.is_ml = args.model.base_model_name in MLModelNames
    args.model.use_class_weighted_loss = (
        args.model.use_class_weighted_loss
        if len(list(args.data.class_names)) > 1
        else False
    )
    return _configure_model_backbone(args)


def _configure_model_backbone(args: Args) -> Args:
    model_cfg = args.model

    if not isinstance(model_cfg, DLModelArgs):
        return args

    if not model_cfg.backbone:
        return args

    if model_cfg.backbone in [
        BackboneNames.ROBERTA_LARGE,
        BackboneNames.XLM_ROBERTA_LARGE,
    ]:
        resolved_backbone = (
            BackboneNames.ROBERTA_LARGE
            if args.data.text_language == DatasetLanguage.ENGLISH
            else BackboneNames.XLM_ROBERTA_LARGE
        )
        if model_cfg.backbone != resolved_backbone:
            logger.info(f'{args.data.text_language}. Switching to {resolved_backbone}')
        model_cfg.backbone = resolved_backbone
    else:
        raise ValueError('Need to handle other backbones explicitly')

    model_cfg.text_dim = model_cfg.get_text_dim(model_cfg.backbone)
    logger.info(f'Setting model.text_dim to {model_cfg.text_dim}')

    if isinstance(model_cfg, RoberteyeArgs):
        if model_cfg.backbone == BackboneNames.ROBERTA_LARGE:
            model_cfg.vocab_size = 50266
        elif model_cfg.backbone == BackboneNames.XLM_ROBERTA_LARGE:
            model_cfg.vocab_size = 250003
        else:
            raise ValueError('Unsupported backbone for RoBERTeye model')
        logger.info(
            f'Setting vocab_size to {model_cfg.vocab_size} for {model_cfg.backbone}'
        )
    return args


def update_cfg_with_wandb(cfg: Args) -> Args:
    """
    Update the configuration object with the wandb config.
    This function will overwrite the config with the wandb config if the
    wandb config is not empty.
    Args:

        cfg (Args): The configuration object.
    Returns:
        Args: The updated configuration object.
    """

    logger.info('Overwriting args with wandb config')

    def validate_and_setattr(
        obj: object, attr_name: str, value: str | int | float
    ) -> None:
        if not hasattr(obj, attr_name):
            raise AttributeError(
                f"Attribute '{attr_name}' does not exist in {obj.__class__.__name__}",
            )
        setattr(obj, attr_name, value)

    for key, value in wandb.config.items():
        if isinstance(value, dict):
            if not hasattr(cfg, key):
                raise AttributeError(
                    f"Attribute '{key}' does not exist in {cfg.__class__.__name__}",
                )
            sub_cfg = getattr(cfg, key)

            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    if not hasattr(sub_cfg, sub_key):
                        raise AttributeError(
                            f"Attribute '{sub_key}' does not exist in {key}",
                        )
                    sub_sub_cfg = getattr(sub_cfg, sub_key)

                    for sub_sub_key, sub_sub_value in sub_value.items():
                        logger.info(
                            f'Setting cfg.{key}.{sub_key}.{sub_sub_key} to {sub_sub_value}',
                        )
                        validate_and_setattr(sub_sub_cfg, sub_sub_key, sub_sub_value)
                else:
                    logger.info(f'Setting cfg.{key}.{sub_key} to {sub_value}')
                    validate_and_setattr(sub_cfg, sub_key, sub_value)
        else:
            logger.info(f'Setting cfg.{key} to {value}')
            validate_and_setattr(cfg, key, value)

    return _configure_model_backbone(cfg)


def setup_logger(
    wandb_project: str,
    wandb_entity: str,
    wandb_job_type: str,
) -> WandbLogger:
    return WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        job_type=wandb_job_type,
    )


def setup_callbacks(
    dir_path: Path,
    early_stopping_patience: int | None,
    max_time: None | str | timedelta | dict[str, int],
    optimize_for_loss: bool,
) -> list[pl_callbacks.Callback]:
    if optimize_for_loss:
        monitor_metric = 'loss/val_all'
        mode = 'min'
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            mode=mode,
            filename='{epoch:02d}-lowest_loss_val_all-{loss/val_all:.4f}',
            dirpath=dir_path,
            auto_insert_metric_name=False,
            verbose=True,
            enable_version_counter=False,
        )
    else:
        raise NotImplementedError(
            'Only optimize_for_loss=True is currently implemented!',
        )
        monitor_metric = 'Classless_Accuracy/val_best_epoch_weighted_average'
        mode = 'max'
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            mode=mode,
            filename='{epoch:02d}-highest_classless_accuracy_val_weighted_average-'
            '{Classless_Accuracy/val_best_epoch_weighted_average:.4f}',
            dirpath=dir_path,
            auto_insert_metric_name=False,
            verbose=True,
            enable_version_counter=False,
        )

    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='step')

    model_summary = pl_callbacks.RichModelSummary(max_depth=4)

    timer = pl_callbacks.Timer(duration=max_time, interval='step')

    checkpoints = [
        checkpoint_callback,
        lr_monitor,
        model_summary,
        # pl_callbacks.RichProgressBar(),
        timer,
    ]

    if early_stopping_patience:
        earlystopping = pl_callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            mode=mode,
        )
        checkpoints.append(earlystopping)
    return checkpoints


def configure_trainer(
    args: TrainerDL,
    work_dir: str | None = None,
    wandb_logger: WandbLogger | None = None,
    accumulate_grad_batches: int = 1,
    max_time: None | str | timedelta | dict[str, int] = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    is_training: bool | None = None,
) -> pl.Trainer:
    precision: Precision = args.precision
    log_every_n_steps = 250
    devices = args.devices
    detect_anomaly = False
    limit_test_batches = None
    limit_val_batches = None
    limit_train_batches = None
    limit_predict_batches = None
    num_sanity_val_steps = 0
    fast_dev_run = False

    if args.run_mode == RunModes.DEBUG:
        max_epochs = 1
        limit_train_batches = 5
        limit_val_batches = 5
        limit_test_batches = 5
        limit_predict_batches = 5
        num_sanity_val_steps = 2
        detect_anomaly = True
        log_every_n_steps = 5
    elif args.run_mode == RunModes.FAST_DEV_RUN:
        fast_dev_run = True

    if is_training is None:
        is_training = work_dir is not None

    callbacks: list[pl_callbacks.Callback] = []
    if is_training:
        if work_dir is None:
            raise ValueError(
                'work_dir must be provided when configuring a training trainer',
            )
        callbacks = setup_callbacks(
            dir_path=Path(work_dir),
            early_stopping_patience=early_stopping_patience,
            max_time=max_time,
            optimize_for_loss=args.optimize_for_loss,
        )

    return pl.Trainer(
        precision=precision,
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        profiler=args.profiler,
        # devices=2,
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        logger=wandb_logger,
        detect_anomaly=detect_anomaly,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_predict_batches=limit_predict_batches,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=True,
    )


def get_checkpoint_path(search_path: Path, checkpoint_template: str) -> Path:
    checkpoint_files = list(search_path.glob(checkpoint_template))

    if not checkpoint_files:
        raise FileNotFoundError(
            f'No checkpoint files found for pattern {checkpoint_template} in {search_path}!',
        )

    logger.info([f.name for f in checkpoint_files])
    # Extract version numbers and sort the list in descending order
    # this is a hacky way to get the version number from the file name
    # Extract highest_classless_accuracy_val_average- values and sort the list in descending order
    pattern = re.compile(
        # r'highest_classless_accuracy_val_weighted_average-(\d+\.\d+)(-v\d+)?\.ckpt$',
        r'lowest_loss_val_all-(\d+\.\d+)(-v\d+)?\.ckpt$',
    )
    mode = 'max' if 'highest' in checkpoint_template else 'min'
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda f: float(
            re.search(pattern, str(f.name)).group(1)
            if re.search(pattern, str(f.name))
            else 0.0
        ),  # type: ignore # noqa: E501
        reverse=mode == 'max',
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            f'No checkpoint files found for pattern {checkpoint_template}!',
        )
    return checkpoint_files[0]


def extract_trial_info(dataset: ETDataset, cols_to_keep: list[str]) -> pd.DataFrame:
    trial_infos = []
    for grouped_data_key in tqdm(dataset.ordered_key_list, desc='Label'):
        trial_info = dataset.grouped_ia_data.get_group(grouped_data_key).iloc[0]
        trial_infos.append(trial_info[cols_to_keep])
    try:
        return pd.DataFrame(trial_infos)
    except pd.errors.InvalidIndexError:
        # remove duplicate columns
        return pd.DataFrame(
            [
                trial_info.loc[~trial_info.index.duplicated()]
                for trial_info in trial_infos
            ]
        )


def convert_string_to_list(s: pd.Series) -> list[list[float]]:
    """
    Converts a Pandas Series containing stringified lists into actual lists.

    Args:
        s (pd.Series): Series with stringified lists.

    Returns:
        list[list[float]]: List of lists with float values.
    """
    return s.apply(ast.literal_eval).tolist()


def plot_grouped_accuracy(
    res,
    group_var,
    remove_legend: bool = False,
    figsize=None,
    continuous: bool = False,
    xlim=None,
):
    # Compute the average accuracy and error bar for each level
    accuracy_by_x = (
        res.groupby([group_var, 'fold_index', 'eval_regime'])['is_correct']
        .mean()
        .reset_index()
    )

    # Plot the average accuracy with error bars for each level
    if figsize:
        plt.figure(figsize=figsize)
    if continuous:
        sns.lineplot(data=accuracy_by_x, x=group_var, hue='eval_regime', y='is_correct')
    else:
        sns.barplot(data=accuracy_by_x, x='eval_regime', hue=group_var, y='is_correct')

    plt.xlabel(group_var)
    plt.ylabel('Average Accuracy')
    plt.title(f'Average Accuracy by {group_var}')

    if xlim:
        plt.xlim(xlim)

    if remove_legend:
        plt.legend().remove()

    plt.show()


def plot_average_roc_curves_with_error_bands(
    model_dfs,
    base_path='figures',
):
    # Get unique evaluation regimes from the first model's data
    first_model_name = next(iter(model_dfs))
    unique_eval_regimes = model_dfs[first_model_name]['eval_regime'].unique()
    num_regimes = len(unique_eval_regimes)

    # Create subplots
    fig, axes = plt.subplots(
        1, num_regimes, figsize=(6 * num_regimes, 6), sharex=True, sharey=True
    )

    # Iterate over each evaluation regime
    for i, regime in enumerate(unique_eval_regimes):
        ax = axes[i] if num_regimes > 1 else axes

        # Iterate over each model
        for model_name, model_res in model_dfs.items():
            # Filter the DataFrame for the current eval_regime
            regime_res = model_res[model_res['eval_regime'] == regime]
            unique_folds = regime_res['fold_index'].unique()
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for fold in unique_folds:
                # Filter the DataFrame for the current fold
                fold_res = regime_res[regime_res['fold_index'] == fold]

                # Compute the false positive rate, true positive rate, and thresholds
                fpr, tpr, _ = roc_curve(fold_res['label'], fold_res['prediction_prob'])

                # Compute the AUROC score
                auroc = roc_auc_score(fold_res['label'], fold_res['prediction_prob'])
                aucs.append(auroc)

                # Interpolate the TPRs to the mean FPRs
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

            # Compute mean and standard deviation of TPRs and AUROC values
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            model_name_mapping = {
                'label': 'Ground Truth Label',
                'LRAvgDWELL': 'Reading Time',
                'A_LRAvgDWELL': 'Reading Time',
                'RoBERTa-QEye-W': 'RoBERTEye-W',
                'RoBERTa-QEye-F': 'RoBERTEye-F',
                'FSE': 'BEyeLSTM - NT',
                'LRDiane': 'Log. Regression',
                'PLMAS': 'PLM-AS',
                'PostFusion': 'PostFusion-Eye',
                'MAG': 'MAG-Eye',
                'Dummy': 'Majority Class',
            }
            # Plot the mean ROC curve

            line_color = (
                'black'
                if model_name_mapping.get(model_name, model_name) == 'Majority Class'
                else None
            )
            # # choose a different color for "Reading Time"
            # line_color = 'darkgreen' if model_name_mapping.get(model_name, model_name) == "Reading Time" else line_color
            line_color = (
                'dodgerblue'
                if model_name_mapping.get(model_name, model_name) == 'RoBERTEye-F'
                else line_color
            )
            ax.plot(
                mean_fpr,
                mean_tpr,
                lw=2,
                alpha=0.8,
                label=f'{model_name_mapping.get(model_name, model_name)} ({mean_auc:.2f} ± {std_auc:.2f})',
                color=line_color,
            )

            # Plot the standard deviation as a shaded area
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                alpha=0.1,
            )

        # Plot the random guess line
        # ax.plot([0, 1], [0, 1], linestyle="--", color="black")

        # Add labels and title
        ax.set_xlabel('False Positive Rate', fontsize=14)
        # if i == 0:
        ax.set_ylabel('True Positive Rate', fontsize=14)

        regime_map = {
            SetNames.SEEN_SUBJECT_UNSEEN_ITEM: 'New Item',
            SetNames.UNSEEN_SUBJECT_SEEN_ITEM: 'New Participant',
            SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM: 'New Item and participant',
        }
        ax.set_title(f'{regime_map[regime]}', fontsize=14)

        # Add legend for each subplot
        ax.legend(loc='lower right')

    # increase tick font size
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # plt.suptitle("ROC Curves by Model and Evaluation Regime")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f'Saving figure to {base_path / "ROC_Curves_by_Model_and_Evaluation_Regime.pdf"}'
    )
    plt.savefig(base_path / 'ROC_Curves_by_Model_and_Evaluation_Regime.pdf')
    plt.show()


def confusion_matrix_by_regime(
    res, unique_eval_regimes=['new_item', 'new_subject', 'new_item_and_subject']
) -> None:
    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    fig.suptitle('Confusion Matrices by Evaluation Regime')
    for i, regime in enumerate(unique_eval_regimes):
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res['eval_regime'] == regime]

        # Compute the confusion matrix
        cm = confusion_matrix(
            regime_res['label'], regime_res['binary_prediction'], normalize='true'
        )

        # Create a heatmap using seaborn in the corresponding subplot
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=['Gathering', 'Hunting'],
            yticklabels=['Gathering', 'Hunting'],
            ax=axes[i],
        )
        axes[i].set_title(f'{regime}')
        axes[i].set_ylabel('Actual Label')
        axes[i].set_xlabel('Predicted Label')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def classification_report_by_regime(
    res, unique_eval_regimes=['new_item', 'new_subject', 'new_item_and_subject']
) -> None:
    for regime in unique_eval_regimes:
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res['eval_regime'] == regime]
        logger.info(f'Classification_report {regime}')
        logger.info(
            classification_report(
                regime_res['label'],
                regime_res['binary_prediction'],
                target_names=['Gathering', 'Hunting'],
            )
        )


def fold_level_roc_curve(
    res, unique_eval_regimes=['new_item', 'new_subject', 'new_item_and_subject']
) -> None:
    # Assuming 'res' is a DataFrame that contains the 'fold_index' column
    unique_folds = res['fold_index'].unique()
    for i, regime in enumerate(unique_eval_regimes):
        # Filter the DataFrame for the current eval_regime
        regime_res = res[res['eval_regime'] == regime]
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=(8, 8))

        for fold in unique_folds:
            # Filter the DataFrame for the current fold
            fold_res = regime_res[regime_res['fold_index'] == fold]

            # Compute the false positive rate, true positive rate, and thresholds
            fpr, tpr, _ = roc_curve(fold_res['label'], fold_res['prediction_prob'])

            # Compute the AUROC score
            auroc = roc_auc_score(fold_res['label'], fold_res['prediction_prob'])
            aucs.append(auroc)

            # Interpolate the TPRs to the mean FPRs
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            # Plot the AUROC curve for the current fold
            plt.plot(
                fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUROC = {auroc:.2f})'
            )

        # Compute mean and standard deviation of TPRs and AUROC values
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Plot the mean ROC curve
        plt.plot(
            mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        # Plot the standard deviation as a shaded area
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color='grey',
            alpha=0.2,
            label=r'$\pm$ 1 std. dev.',
        )

        # Plot the random guess line
        plt.plot([0, 1], [0, 1], linestyle='--', color='r')

        # Add labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC Curves for {regime}')

        # Add legend
        plt.legend(loc='lower right')
        plt.show()


def prepare_data(model_df, params, re_cols, outcome):
    model_df_input = model_df[params + re_cols + [outcome]].copy()
    return model_df_input


def standardize_features(model_df_input, params, with_std):
    scaler = StandardScaler(with_std=with_std)
    model_df_input[params] = scaler.fit_transform(model_df_input[params])
    logger.info(f'Standardized coefficients with_std={with_std}')
    return model_df_input


def construct_formula(outcome, params, random_effects_structure):
    concatenated_params = '+'.join(params)

    if random_effects_structure == 'full':
        formula = f'{outcome} ~ {concatenated_params} + (1 + {concatenated_params} | participant_id) + (1 + {concatenated_params} | unique_paragraph_id) + (1 + {concatenated_params} | eval_regime)'
    elif random_effects_structure == 'nested':
        random_effects_subject = ' + '.join(
            [f'(1 + {param} | participant_id)' for param in params]
        )
        random_effects_paragraph = ' + '.join(
            [f'(1 + {param} | unique_paragraph_id)' for param in params]
        )
        random_effects_eval_regime = ' + '.join(
            [f'(1 + {param} | eval_regime)' for param in params]
        )
        formula = f'{outcome} ~ {concatenated_params} + {random_effects_subject} + {random_effects_paragraph} + {random_effects_eval_regime}'
    elif random_effects_structure == 'crossed':
        random_effects_subject = ' + '.join(
            [f'(1 | participant_id) + ({param} | participant_id)' for param in params]
        )
        random_effects_paragraph = ' + '.join(
            [
                f'(1 | unique_paragraph_id) + ({param} | unique_paragraph_id)'
                for param in params
            ]
        )
        random_effects_eval_regime = ' + '.join(
            [f'(1 | eval_regime) + ({param} | eval_regime)' for param in params]
        )
        formula = f'{outcome} ~ {concatenated_params} + {random_effects_subject} + {random_effects_paragraph} + {random_effects_eval_regime}'
    elif random_effects_structure == 'intercept':
        formula = f'{outcome} ~ {concatenated_params} + (1 | participant_id) + (1 | unique_paragraph_id)'  # + (1 | eval_regime)"
    else:
        raise ValueError('Invalid random effects structure')
    logger.info(f'Random effects structure: {random_effects_structure}')
    logger.info(f'Formula: {formula}')
    return formula


def get_outcome_variable(is_correct_05):
    if is_correct_05:
        outcome = 'is_correct_05'
        logger.info('Using is_correct_05 as outcome.')
    else:
        outcome = 'is_correct'
        logger.info('Using is_correct as outcome.')
    return outcome


def remove_nan_values(model_df_input):
    if model_df_input.isnull().values.any():
        nan_cols = model_df_input.columns[model_df_input.isnull().any()].tolist()
        total_rows_before = model_df_input.shape[0]
        model_df_input = model_df_input.dropna()
        total_rows_after = model_df_input.shape[0]
        rows_removed = total_rows_before - total_rows_after
        logger.info(f'Dropped NaN values coming from columns: {nan_cols}')
        logger.info(
            f'Removed {rows_removed} ({rows_removed / total_rows_before:.2%}%) rows out of {total_rows_before} total rows.'
        )
    return model_df_input


def map_pvalue_to_asterisks(pvalue: float) -> str:
    if pvalue <= 0.001:
        return '***'
    elif pvalue <= 0.01:
        return '**'
    elif pvalue <= 0.05:
        return '*'
    else:
        return 'n.s.'


def get_config(config_path: Path) -> Args:
    """
    Load the config for testing.
    """
    output_dir = to_absolute_path(str(config_path))
    overrides = OmegaConf.load(join(output_dir, '.hydra/overrides.yaml'))
    hydra_config = OmegaConf.load(join(output_dir, '.hydra/hydra.yaml'))

    # getting the config name from the previous job.
    config_name = hydra_config.hydra.job.config_name

    # compose a new config from scratch
    cfg = compose(config_name, overrides=overrides)
    updated_cfg = instantiate(cfg, _convert_='object')

    return updated_cfg


def defaultdict_to_df(
    macro_auroc: dict[str, list[float]],
    binary_auroc: dict[str, list[float]],
) -> pd.DataFrame:
    # Convert the default dicts to DataFrames
    df_macro = pd.DataFrame(macro_auroc.items(), columns=['Eval Type', 'AUROC'])
    df_binary = pd.DataFrame(binary_auroc.items(), columns=['Eval Type', 'AUROC'])

    # Add a new column to distinguish between macro and binary
    df_macro['Task'] = 'macro'
    df_binary['Task'] = 'binary'

    # Concatenate the DataFrames
    df = pd.concat([df_macro, df_binary])

    # Calculate the average and standard deviation of AUROC
    df['Average AUROC'] = df['AUROC'].apply(np.mean)
    df['STD AUROC'] = df['AUROC'].apply(np.std)

    # # Set the index
    df.set_index(['Task', 'Eval Type'], inplace=True)
    return df


def raw_res_to_auroc(res: pd.DataFrame) -> pd.DataFrame:
    grouped_res = res.groupby(['eval_type', 'fold_index'])
    macro_auroc = defaultdict(list)
    binary_auroc = defaultdict(list)
    for (eval_type, fold_index), group_data in grouped_res:
        labels = group_data['label'].tolist()
        preds = convert_string_to_list(group_data['prediction_prob'])
        macro_auroc[eval_type].append(
            round(
                roc_auc_score(
                    y_true=labels,
                    y_score=preds,
                    average='macro',
                    multi_class='ovr',
                ),
                3,
            ),
        )

        binary_labels = group_data['binary_label'].tolist()
        binary_preds = group_data['binary_prediction_prob'].tolist()
        binary_auroc[eval_type].append(
            round(roc_auc_score(y_true=binary_labels, y_score=binary_preds), 3),
        )

    return defaultdict_to_df(macro_auroc, binary_auroc)
