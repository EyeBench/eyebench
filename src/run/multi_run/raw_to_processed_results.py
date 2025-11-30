from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.configs.constants import (
    REGIMES,
    DiscriSupportedMetrics,
    RegrSupportedMetrics,
    SetNames,
)
from src.configs.data import DATA_CONFIGS_MAPPING
from src.run.single_run.utils import convert_string_to_list

ALL_REGIMES = REGIMES + ['all']

# Regression tasks (tasks where the target is continuous)
REG_TASKS = [
    'MECOL2_LEX',  # Vocabulary Knowledge
    'CopCo_RCS',  # Reading Comprehension Skill
    'SBSAT_STD',  # Subjective Text Difficulty
]


def load_trial_level_test_results(
    file_path: Path,
    on_error='raise',
    verbose=False,
) -> pd.DataFrame | None:
    try:
        res = pd.read_csv(file_path)
    except FileNotFoundError as e:
        if verbose:
            logger.warning(f'File not found: {file_path}')
        if on_error == 'raise':
            raise e
        return None
    return res


def _process_prediction_prob(metric_name, prediction_prob: pd.Series) -> pd.Series:
    """from prediction_prob to y_pred"""
    if metric_name == DiscriSupportedMetrics.BALANCED_ACCURACY:
        y_pred = (prediction_prob > 0.5).astype(int)
    else:
        y_pred = prediction_prob
    return y_pred


def get_scores(y_true, prediction_prob, metric_name: str, **kwargs):
    result = None

    y_pred = _process_prediction_prob(metric_name, prediction_prob)

    if metric_name == DiscriSupportedMetrics.BALANCED_ACCURACY:
        result = balanced_accuracy_score(y_true, y_pred, **kwargs)
    elif metric_name == DiscriSupportedMetrics.AUROC:
        result = roc_auc_score(y_true, y_pred, **kwargs)
    elif metric_name == RegrSupportedMetrics.RMSE:
        result = root_mean_squared_error(y_true, y_pred, **kwargs)
    elif metric_name == RegrSupportedMetrics.MAE:
        result = mean_absolute_error(y_true, y_pred, **kwargs)
    elif metric_name == RegrSupportedMetrics.R2:
        result = r2_score(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f'Unknown metric name: {metric_name}')

    if np.isnan(result):
        # raise ValueError(f'Metric {metric_name} returned NaN for the given inputs')
        logger.warning(f'Metric {metric_name} returned NaN for the given inputs')
    return float(result)


def validate_results(
    eval_type: str,
    data_task: str,
    model: str,
    eval_regime,
    metric_type: str,
    metric_name,
    y_true: pd.Series,
    prediction_prob: pd.Series,
    fold,
    n_folds: int,
    fold_list,
) -> dict:
    """Validate the results before computing statistics."""

    # Process prediction probabilities to get predicted classes/values
    y_pred = _process_prediction_prob(metric_name, prediction_prob)

    unique_y_true = np.unique(y_true)
    unique_y_pred = np.unique(y_pred)

    # If classification, ensure predicted classes are a subset of true classes
    if metric_type != 'Regression':
        extra_classes = set(unique_y_pred) - set(unique_y_true)
        # convert to float and round for readability
        extra_classes = [round(float(x), 2) for x in extra_classes]
        if len(extra_classes) > 5:
            extra_classes = (
                list(extra_classes)[:5]
                + ['...']
                + [f'N extra classes: {len(extra_classes)}']
            )

    # get distribution if n_unique < 10 else ['Too many classes: {n_unique}']
    y_true_distribution = (
        y_true.value_counts().to_dict()
        if len(unique_y_true) < 10
        else {f'N unique: {len(unique_y_true)}'}
    )
    y_pred_distribution = (
        y_pred.value_counts().to_dict()
        if len(unique_y_pred) < 10
        else {f'N unique: {len(unique_y_pred)}'}
    )

    stats = {
        'metric_type': metric_type,
        'metric_name': metric_name.value,
        'eval_type': eval_type,
        'data_task': data_task,
        'model': model,
        'eval_regime': eval_regime,
        'fold': fold,
        'n_folds': n_folds,
        'fold_list': fold_list,
        'n_samples': len(y_true),
        'n_unique_y_true': len(np.unique(y_true)),
        'n_unique_y_pred': len(np.unique(y_pred)),
        'y_true_single': len(np.unique(y_true)) == 1,
        'y_pred_single': len(np.unique(y_pred)) == 1,
        'y_true_distribution': y_true_distribution,
        'y_pred_distribution': y_pred_distribution,
        'extra_classes_in_y_pred': list(extra_classes)
        if metric_type != 'Regression' and extra_classes
        else None,
    }

    return stats


def get_metric_from_raw_res(
    res: pd.DataFrame,
    metric_type: object,
    metric_name: str,
    data_task: str,
    model: str,
    eval_type: str,
) -> pd.DataFrame:
    """
    Converts raw results dataframe into accuracy metrics by evaluation regime.

    Args:
        res: DataFrame containing predictions and labels
        metric_name: Metric to compute ('accuracy' or 'balanced_accuracy')

    Returns:
        DataFrame with metrics for each evaluation regime and overall
    """
    res_df = defaultdict(list)
    stats = []

    # Calculate metrics for each evaluation regime and fold
    for (eval_regime, fold), group in res.groupby(['eval_regime', 'fold_index']):
        y_true = group['label']
        prediction_prob = group['prediction_prob']
        n_folds = group['fold_index'].nunique()
        fold_list = group['fold_index'].unique().tolist()

        new_stats = validate_results(
            eval_type,
            data_task,
            model,
            eval_regime,
            metric_type,
            metric_name,
            y_true,
            prediction_prob,
            fold,
            n_folds,
            fold_list,
        )

        score = get_scores(
            y_true=y_true,
            prediction_prob=prediction_prob,
            metric_name=metric_name,
        )
        res_df[eval_regime].append(score)
        new_stats['score'] = score
        stats.append(new_stats)

    # Calculate overall metrics for each fold
    for fold, group in res.groupby('fold_index'):
        y_true = group['label']
        prediction_prob = group['prediction_prob']
        eval_regime = 'all'
        n_folds = group['fold_index'].nunique()
        fold_list = group['fold_index'].unique().tolist()

        new_stats = validate_results(
            eval_type,
            data_task,
            model,
            eval_regime,
            metric_type,
            metric_name,
            y_true,
            prediction_prob,
            fold,
            n_folds,
            fold_list,
        )

        score = get_scores(
            y_true=y_true,
            prediction_prob=prediction_prob,
            metric_name=metric_name,
        )
        res_df[eval_regime].append(score)
        new_stats['score'] = score
        stats.append(new_stats)

    # Calculate overall metrics for each evaluation regime
    for eval_regime, group in res.groupby('eval_regime'):
        y_true = group['label']
        prediction_prob = group['prediction_prob']
        fold = 'all'
        n_folds = group['fold_index'].nunique()
        fold_list = group['fold_index'].unique().tolist()

        new_stats = validate_results(
            eval_type,
            data_task,
            model,
            eval_regime,
            metric_type,
            metric_name,
            y_true,
            prediction_prob,
            fold,
            n_folds,
            fold_list,
        )

        score = get_scores(
            y_true=y_true,
            prediction_prob=prediction_prob,
            metric_name=metric_name,
        )
        new_stats['score'] = score
        stats.append(new_stats)

    # Create DataFrame and ensure all regimes are included
    df = pd.DataFrame(
        res_df.items(),
        columns=['Eval Regime', metric_name],
    ).set_index('Eval Regime')
    # Concatenate all stats DataFrames
    stats_df = pd.DataFrame(stats)
    # Reindex to ensure all regimes are present, even if empty
    return df.reindex(ALL_REGIMES), stats_df


def aggregate_df(
    res_df: pd.DataFrame,
    metric_name: str,
    metric_type: str,
    columns: list[str] = ALL_REGIMES,
    error_type: str = 'std',
) -> pd.DataFrame:
    """
    Aggregates metrics data by computing mean and error statistics.

    Args:
        res_df: DataFrame containing lists of metric values for each regime
        metric_name: Name of the metric (e.g., 'accuracy', 'balanced_accuracy')
        columns: List of evaluation regimes to process
        error_type: Type of error to calculate ('std' or 'sem')

    Returns:
        DataFrame with aggregated metrics and formatted results
    """
    res_df = res_df.copy()

    for col in columns:
        # Calculate mean values and convert to percentage rounded to 1 decimal place
        avg_col = f'Avg {metric_name} {col}'
        res_df[avg_col] = res_df[col].apply(np.nanmean)

        # if metric_type is not 'Regression':
        if metric_type == 'Regression':
            res_df[avg_col] = res_df[avg_col].round(2)
        else:
            res_df[avg_col] = (100 * res_df[avg_col]).round(1)

        def handle_missing(x):
            if isinstance(x, list):
                if len(x) > 0:
                    return np.std(x) / np.sqrt(len(x))
            return 0

        # Calculate error statistics based on specified type
        if error_type == 'std':
            err_col = f'Std {metric_name} {col}'
            res_df[err_col] = res_df[col].apply(np.nanstd)
        elif error_type == 'sem':
            err_col = f'SEM {metric_name} {col}'
            res_df[err_col] = res_df[col].apply(lambda x: handle_missing(x))
        else:
            raise ValueError(f'Invalid error type: {error_type}')

        # Convert error to percentage and round
        if metric_type == 'Regression':
            res_df[err_col] = res_df[err_col].round(1)
        else:
            res_df[err_col] = (100 * res_df[err_col]).round(1)

        # Format the final result string with mean ± error
        formatted_col = f'{col}'.replace('_', ' ').capitalize()
        res_df[formatted_col] = res_df.apply(
            lambda x: f'{x[avg_col]} ± {x[err_col]}',
            axis=1,
        )

        # Remove intermediate columns
        res_df = res_df.drop([col, avg_col, err_col], axis=1)

    return res_df


def compute_statistics(
    tasks: list[str],
    models: dict[str, dict[str, str]],
    results_dir: Path,
    results_raw_dir: Path,
) -> None:
    all_res = collect_results_from_folds(tasks, models, results_raw_dir)
    save_metric_to_csv(all_res=all_res, results_dir=results_dir, models=models)


SUPPORTED_METRICS = {
    'Discriminative': DiscriSupportedMetrics,
    'Regression': RegrSupportedMetrics,
}


def collect_results_from_folds(tasks, models, results_raw_dir):
    # Initialize a dictionary to store results
    all_res = defaultdict(dict)

    for task in tasks:
        if task not in DATA_CONFIGS_MAPPING:
            raise ValueError(
                f'Task {task} not found in DATA_CONFIGS_MAPPING. Options are: {list(DATA_CONFIGS_MAPPING.keys())}'
            )

        # Get n_folds from the data config
        data_config_class = DATA_CONFIGS_MAPPING[task]()
        n_folds = data_config_class.n_folds

        # Determine if this is a regression task
        is_regression_task = task in REG_TASKS

        for model, model_info in models.items():
            # Get model capabilities from model dict
            supports_regression = model_info.get('is_regression', False)
            supports_classification = model_info.get('is_classification', False)

            # Skip if model doesn't support the task type
            if is_regression_task and not supports_regression:
                continue

            if not is_regression_task and not supports_classification:
                continue

            trainer = model_info['trainer']
            fold_res: list[pd.DataFrame] = []
            found_fold_indices: list[int] = []
            for fold_index in range(n_folds):
                file_path = (
                    results_raw_dir
                    / f'+data={task},+model={model},+trainer={trainer},trainer.wandb_job_type={model_info["model"]}_{task}'
                    / f'{fold_index=}'
                    / 'trial_level_test_results.csv'
                )
                df = load_trial_level_test_results(
                    on_error='continue', file_path=file_path
                )
                if df is not None:
                    fold_res.append(df)
                    found_fold_indices.append(fold_index)

            if not fold_res:
                logger.warning(
                    f'{task} - {model} - {trainer} not found | Path: {file_path}'
                )
                continue

            logger.info(
                f'Found {task} - {model} - {trainer} with {len(fold_res)} folds ({found_fold_indices})'
            )
            res = pd.concat(fold_res)

            # Process the prediction probabilities
            if res['prediction_prob'].dtype == 'object':
                # Convert string representation of lists to actual lists
                preds = convert_string_to_list(res['prediction_prob'])
                preds = np.array(preds)
                if len(preds.squeeze().shape) != 1:
                    # Get the second column for binary classification
                    preds = preds[:, 1]

                res['prediction_prob'] = preds.squeeze()

            # if task == 'OneStop_RC':
            #     for combination in (
            #         res[['repeated_reading_trial', 'question_preview']]
            #         .drop_duplicates()
            #         .values
            #     ):
            #         repeated_reading_trial, question_preview = combination
            #         sub_res = res.loc[
            #             (res['repeated_reading_trial'] == repeated_reading_trial)
            #             & (res['question_preview'] == question_preview)
            #         ]
            #         r = 'Repeated' if repeated_reading_trial else 'First'
            #         p = 'Preview' if question_preview else 'NoPreview'
            #         all_res[f'{task}_{r}_{p}'][model] = sub_res

            # Store the results in the dictionary
            all_res[task][model] = res

    return all_res


def save_metric_to_csv(all_res, results_dir, models):
    stats_dfs = []
    for metric_type, curr_supported_metrics in SUPPORTED_METRICS.items():
        for metric_name in curr_supported_metrics:
            metrics = []
            for data_task_, data_task_res in all_res.items():
                # Determine if this is a regression task
                is_regression_task = data_task_ in REG_TASKS or any(
                    reg_task in data_task_ for reg_task in REG_TASKS
                )

                for model, res in data_task_res.items():
                    # Get model capabilities from model dict
                    supports_regression = models[model].get('is_regression', False)
                    supports_classification = models[model].get(
                        'is_classification', False
                    )

                    # Skip if model doesn't support the task type
                    if (
                        is_regression_task
                        and (
                            not supports_regression
                            or metric_name not in RegrSupportedMetrics
                        )
                    ) or (
                        not is_regression_task
                        and (
                            not supports_classification
                            or metric_name not in DiscriSupportedMetrics
                        )
                    ):
                        continue
                    for eval_type in [SetNames.VAL, SetNames.TEST]:
                        res_val = res[res['eval_type'] == eval_type]
                        if res_val['prediction_prob'].isna().any():
                            raise ValueError(
                                f'NaN values found in {data_task_} - {model} - {eval_type} in prediction_prob'
                            )

                        metric_val, stats_df = get_metric_from_raw_res(
                            res=res_val,
                            metric_type=metric_type,
                            metric_name=metric_name,
                            data_task=data_task_,
                            model=model,
                            eval_type=eval_type,
                        )
                        metric_val = metric_val[[metric_name]].T
                        metric_val.index = pd.MultiIndex.from_tuples(
                            [(model, data_task_)], names=['Model', 'Data']
                        )
                        metric_val['Eval Type'] = eval_type
                        metrics.append(metric_val)
                        stats_dfs.append(stats_df)

            # if not metrics continue
            if not metrics:
                continue

            # Concatenate the metric values into a single DataFrame
            res_df = (
                pd.concat(metrics)
                # .swaplevel()
                .reset_index()
                .set_index(['Eval Type', 'Data', 'Model'])
                .sort_index(
                    level=['Eval Type', 'Data', 'Model'], ascending=[False, True, True]
                )
            )

            regime = aggregate_df(
                res_df,
                metric_name,
                metric_type,
                error_type='sem',
            )

            results_dir.mkdir(exist_ok=True, parents=True)
            result_path = results_dir / f'{metric_name}.csv'
            regime.to_csv(result_path)
            res_df.to_csv(results_dir / f'{metric_name}_fold_level.csv')
            logger.info(f'Saved {metric_name} results to {result_path}')

    # Save stats dataframe
    all_stats = pd.concat(stats_dfs, ignore_index=True)
    # save each datatask stats to a separate csv
    for data_task in all_stats['data_task'].unique():
        task_stats = all_stats[all_stats['data_task'] == data_task]
        task_stats_path = results_dir / f'stats_{data_task}.csv'
        task_stats.to_csv(task_stats_path, index=False)
        logger.info(f'Saved results statistics for {data_task} to {task_stats_path}')


if __name__ == '__main__':
    base_results_dir = Path.cwd() / 'results'
    results_save_dir = base_results_dir / 'eyebench_benchmark_results'
    results_raw_dir = base_results_dir / 'raw'
    TrainerDL = {'trainer': 'TrainerDL'}
    TrainerML = {'trainer': 'TrainerML'}
    tasks = [
        'CopCo_RCS',  # Reading Comprehension Skill (Regression)
        'CopCo_TYP',  # Dyslexia Detection
        'MECOL2_LEX',  # Vocabulary Knowledge (Regression)
        'OneStop_RC',  # Reading Comprehension
        'SBSAT_RC',  # Reading Comprehension
        'PoTeC_RC',  # Reading Comprehension
        'SBSAT_STD',  # Subjective Text Difficulty (Regression)
        'PoTeC_DE',  # Domain Expertise
        'IITBHGC_CV',  # Claim Verification
    ]

    # Helper to build model entries
    def _build_entries(names, base, is_regression, is_classification):
        return {
            name: base
            | {
                'model': name,
                'is_regression': is_regression,
                'is_classification': is_classification,
            }
            for name in names
        }

    # Deep Learning models (can handle both classification and regression)
    dl_model_names = [
        'AhnCNN',
        'AhnRNN',
        'BEyeLSTMArgs',
        'MAG',
        'PLMASArgs',
        'PLMASfArgs',
        'PostFusion',
        'Roberta',
        'RoberteyeFixation',
        'RoberteyeWord',
    ]
    dl_models = _build_entries(dl_model_names, TrainerDL, True, True)

    # Classical ML Classification models (classification only)
    ml_classification_names = [
        'SupportVectorMachineMLArgs',
        # 'XGBoostMLArgs',
        'LogisticRegressionMLArgs',
        'LogisticMeziereArgs',
        'DummyClassifierMLArgs',
        'RandomForestMLArgs',
    ]
    ml_classification_models = _build_entries(
        ml_classification_names, TrainerML, False, True
    )

    # Classical ML Regression models (regression only)
    ml_regression_names = [
        'SupportVectorRegressorMLArgs',
        # 'XGBoostRegressorMLArgs',
        'RandomForestRegressorMLArgs',
        'LinearRegressionArgs',
        'LinearMeziereArgs',
        'DummyRegressorMLArgs',
    ]
    ml_regression_models = _build_entries(ml_regression_names, TrainerML, True, False)

    # Combine all models
    models = dl_models | ml_classification_models | ml_regression_models

    compute_statistics(
        tasks=tasks,
        models=models,
        results_dir=results_save_dir,
        results_raw_dir=results_raw_dir,
    )
