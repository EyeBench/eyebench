"""
Convert the metric CSV from the EyeBench benchmark into a LaTeX table.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from loguru import logger

from src.configs.constants import DiscriSupportedMetrics, RegrSupportedMetrics
from src.run.multi_run.raw_to_processed_results import REG_TASKS

# ——— CONSTANTS ———
do_metric_level_tables = False  # todo Make a hyperparamter? delete
#: CSV file with AUROC results
CSV_BASE_PATH = Path('results/eyebench_benchmark_results')

#: Local directory to save formatted results
LOCAL_OUTPUT_DIR = Path('results/formatted_eyebench_benchmark_results')

#: Where to copy the generated .tex files
OVERLEAF_OUTPUT_DIR = Path(
    '~/67eea3c68a5e16fe7932778e/results'
).expanduser()  # TODO change before code release

#: Mapping from dataset key to short column header
DATASET_TO_COLUMN = {
    'CopCo_RCS': '\\textbf{Reading Comprehension Skill} task on \\textbf{CopCo}',
    'CopCo_TYP': '\\textbf{Dyslexia Detection} task on \\textbf{CopCo}',
    'MECOL2_LEX': '\\textbf{Vocabulary Knowledge} task on \\textbf{MECOL2}',
    'OneStop_RC': '\\textbf{Reading Comprehension} task on \\textbf{OneStop}',
    'SBSAT_RC': '\\textbf{Reading Comprehension} task on \\textbf{SBSAT}',
    'PoTeC_RC': '\\textbf{Reading Comprehension} task on \\textbf{PoTeC}',
    'SBSAT_STD': '\\textbf{Subjective Text Difficulty} task on \\textbf{SBSAT}',
    'PoTeC_DE': '\\textbf{Domain Expertise} task on \\textbf{PoTeC}',
    'IITBHGC_CV': '\\textbf{Claim Verification} task on \\textbf{IITBHGC}',
}

#: Groups of tasks, in the order they should appear
GROUPS = {
    'Reader': [
        'MECOL2_LEX',  # Vocabulary Knowledge (continuous)
        'CopCo_RCS',  # Reading Comprehension Skill (continuous)
        'CopCo_TYP',  # Dyslexia Detection
    ],
    r'Reader \& Text': [
        'OneStop_RC',  # Reading Comprehension
        'SBSAT_RC',  # Reading Comprehension
        'PoTeC_RC',  # Reading Comprehension
        'SBSAT_STD',  # Subjective Text Difficulty
        'PoTeC_DE',  # Domain Expertise
        'IITBHGC_CV',  # Claim Verification
    ],
}

REG_TASKS_STR = (
    f'{DATASET_TO_COLUMN["MECOL2_LEX"]} and {DATASET_TO_COLUMN["CopCo_RCS"]}'
)

MODEL_ORDER_CLASSIFICATION = [
    'DummyClassifierMLArgs',
    'LogisticRegressionMLArgs',
    'Roberta',
    'LogisticMeziereArgs',
    'SupportVectorMachineMLArgs',
    # 'XGBoostMLArgs',
    'RandomForestMLArgs',
    'AhnRNN',
    'AhnCNN',
    'BEyeLSTMArgs',
    'PLMASArgs',
    'PLMASfArgs',
    'RoberteyeWord',
    'RoberteyeFixation',
    'MAG',
    'PostFusion',
]

MODEL_ORDER_REGRESSION = [
    'DummyRegressorMLArgs',
    'LinearRegressionArgs',
    'Roberta',
    'LinearMeziereArgs',
    'SupportVectorRegressorMLArgs',
    # 'XGBoostRegressorMLArgs',
    'RandomForestRegressorMLArgs',
    'AhnRNN',
    'AhnCNN',
    'BEyeLSTMArgs',
    'PLMASArgs',
    'PLMASfArgs',
    'RoberteyeWord',
    'RoberteyeFixation',
    'MAG',
    'PostFusion',
]

MODEL_ORDER_BY_METRIC_TYPE = {
    'Regression': MODEL_ORDER_REGRESSION,
    'Classification': MODEL_ORDER_CLASSIFICATION,
}

#: How to render each model in the first column (with citations)
MODEL_TO_COLUMN = {
    'DummyClassifierMLArgs': 'Majority Class / Chance',
    'DummyRegressorMLArgs': 'Majority Class / Chance',
    'LogisticRegressionMLArgs': 'Reading Speed',
    'LinearRegressionArgs': 'Reading Speed',
    'Roberta': 'Text-Only Roberta',
    'LogisticMeziereArgs': 'Logistic Regression~\\cite{meziere2023using}',
    'LinearMeziereArgs': 'Logistic Regression~\\cite{meziere2023using}',
    'SupportVectorMachineMLArgs': 'SVM~\\cite{hollenstein2023zuco}',
    'SupportVectorRegressorMLArgs': 'SVM~\\cite{hollenstein2023zuco}',
    # 'XGBoostMLArgs': 'XGBoost',
    # 'XGBoostRegressorMLArgs': 'XGBoost',
    'RandomForestMLArgs': 'Random Forest~\\cite{makowski2024detection}',
    'RandomForestRegressorMLArgs': 'Random Forest~\\cite{makowski2024detection}',
    'AhnCNN': 'AhnCNN~\\citep{ahn2020towards}',
    'AhnRNN': 'AhnRNN~\\citep{ahn2020towards}',
    'BEyeLSTMArgs': 'BEyeLSTM~\\citep{reich_inferring_2022}',
    'MAG': 'MAG-Eye~\\citep{Shubi2024Finegrained}',
    'PLMASArgs': 'PLM-AS~\\citep{Yang2023PLMASPL}',
    'PLMASfArgs': 'PLM-AS-RM~\\citep{haller2022eye}',
    'RoberteyeWord': 'RoBERTEye-W~\\citep{Shubi2024Finegrained}',
    'RoberteyeFixation': 'RoBERTEye-F~\\citep{Shubi2024Finegrained}',
    'PostFusion': 'PostFusion-Eye~\\citep{Shubi2024Finegrained}',
}

METRICS_LABELS = {
    'auroc': 'AUROC',
    'accuracy': 'Accuracy',
    'balanced_accuracy': '\\makecell{Balanced\\\\Accuracy}',
    'f1': 'F1 Score',
    'rmse': 'RMSE',
    'mae': 'MAE',
    'r2': 'R²',
}

ML_REGRESSION_TO_CLASSIFICATION = {
    'DummyRegressorMLArgs': 'DummyClassifierMLArgs',
    'LinearRegressionArgs': 'LogisticRegressionMLArgs',
    'LinearMeziereArgs': 'LogisticMeziereArgs',
    'RandomForestRegressorMLArgs': 'RandomForestMLArgs',
    'SupportVectorRegressorMLArgs': 'SupportVectorMachineMLArgs',
    # 'XGBoostRegressorMLArgs': 'XGBoostMLArgs',
}

REGIME_COLS = {
    'Unseen subject seen item': 'Unseen Reader',
    'Seen subject unseen item': 'Unseen Text',
    'Unseen subject unseen item': 'Unseen Text \\& Reader',
    'All': 'Average',
}


FEATURE_TYPES = {
    'DummyClassifierMLArgs': {
        'Layout': '-',
        'Saccade/Fixation': '-',
        'Word-Level': '-',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    'LogisticRegressionMLArgs': {
        'Layout': '-',
        'Saccade/Fixation': '-',
        'Word-Level': '-',
        'Trial-Level': '\\checkmark',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    'Roberta': {
        'Layout': '-',
        'Saccade/Fixation': '-',
        'Word-Level': '-',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '\\checkmark',
    },
    'LogisticMeziereArgs': {
        'Layout': '-',
        'Saccade/Fixation': '-',
        'Word-Level': '-',
        'Trial-Level': '\\checkmark',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    'SupportVectorMachineMLArgs': {
        'Layout': '-',
        'Saccade/Fixation': '-',
        'Word-Level': '-',
        'Trial-Level': '\\checkmark',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    # 'XGBoostMLArgs': {
    #     'Layout': '-',
    #     'Saccade/Fixation': '-',
    #     'Word-Level': '-',
    #     'Trial-Level': '\\checkmark',
    #     'Linguistic': '\\checkmark',
    #     'Embeddings': '-',
    # },
    'RandomForestMLArgs': {
        'Layout': '-',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '-',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '-',
    },
    'AhnRNN': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '-',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    'AhnCNN': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '-',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '-',
    },
    'BEyeLSTMArgs': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '\\checkmark',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '-',
    },
    'PLMASArgs': {
        'Layout': '-',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '-',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '\\checkmark',
    },
    'PLMASfArgs': {
        'Layout': '-',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '\\checkmark',
        'Trial-Level': '-',
        'Linguistic': '-',
        'Embeddings': '\\checkmark',
    },
    'RoberteyeWord': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '-',
        'Word-Level': '\\checkmark',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '\\checkmark',
    },
    'RoberteyeFixation': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '\\checkmark',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '\\checkmark',
    },
    'MAG': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '-',
        'Word-Level': '\\checkmark',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '\\checkmark',
    },
    'PostFusion': {
        'Layout': '\\checkmark',
        'Saccade/Fixation': '\\checkmark',
        'Word-Level': '\\checkmark',
        'Trial-Level': '\\checkmark',
        'Linguistic': '\\checkmark',
        'Embeddings': '\\checkmark',
    },
}


# Define task order for the combined table - keep all tasks even if data not available yet
# Classification tasks (AUROC - higher is better)
classification_tasks = [
    'OneStop_RC',  # Reading Comprehension (OneStop)
    'SBSAT_RC',  # Reading Comprehension (SB-SAT)
    'PoTeC_RC',  # Reading Comprehension (PoTeC)
    'PoTeC_DE',  # Domain Expertise (PoTeC)
    'IITBHGC_CV',  # Claim Verification (IITB-HGC)
    'CopCo_TYP',  # Dyslexia Detection (CopCo)
]

# Regression tasks (RMSE - lower is better)
regression_tasks = [
    'SBSAT_STD',  # Subjective Text Difficulty (SB-SAT)
    'CopCo_RCS',  # Reading Comprehension Skill (CopCo)
    'MECOL2_LEX',  # Vocabulary Knowledge (MECO L2)
]

# Task labels for header
task_headers = {
    'OneStop_RC': ('Reading Compr.', 'OneStop'),
    'SBSAT_RC': ('Reading Compr.', 'SB-SAT'),
    'PoTeC_RC': ('Reading Compr.', 'PoTeC'),
    'PoTeC_DE': ('Domain\\\\Expertise', 'PoTeC'),
    'IITBHGC_CV': ('Claim\\\\Verification', 'IITB-HGC'),
    'CopCo_TYP': ('Dyslexia\\\\Detection \\woman{}', 'CopCo'),
    'SBSAT_STD': ('Subj. Text\\\\Difficulty \\woman{}+\\page{}', 'SB-SAT'),
    'CopCo_RCS': ('Reading Comp.\\\\Skill \\woman{}', 'CopCo'),
    'MECOL2_LEX': ('\\newthing{} Vocabulary\\\\Knowledge \\woman{}', 'MECO L2'),
}


def extract_numeric_value(val_str):
    """Extract numeric value from string format like '65.0 ± 0.0'"""
    if pd.isna(val_str) or val_str == '' or val_str == '-':
        return np.nan
    val_str = str(val_str)
    try:
        return float(val_str.split(' ±')[0])
    except (ValueError, IndexError):
        try:
            return float(val_str)
        except ValueError:
            return np.nan


def format_value_with_subscript(value: str) -> str:
    """
    Format a value string to use LaTeX subscript for standard deviation.

    Converts '65.0 ± 2.3' to '65.0\\textsubscript{±2.3}'

    Args:
        value: String in format 'mean ± std' or just a value

    Returns:
        Formatted string with std as subscript
    """
    if not value or value == '' or value == '-':
        return value

    value_str = str(value)
    if ' ± ' in value_str:
        mean, std = value_str.split(' ± ')
        return f'{mean}\\textsubscript{{±{std}}}'
    return value_str


def is_metric_higher_better(metric_name: str) -> bool:
    """
    Determine if higher values are better for a given metric.

    Args:
        metric_name: The metric name (e.g., 'auroc', 'rmse', 'r2', 'R²')

    Returns:
        True if higher is better, False if lower is better
    """
    # Normalize metric name to lowercase for comparison
    metric_lower = metric_name.lower().strip()

    # Metrics where lower is better
    lower_is_better = ['rmse', 'mae']

    # Metrics where higher is better
    higher_is_better = ['auroc', 'accuracy', 'balanced_accuracy', 'f1', 'r2', 'r²']

    if metric_lower in lower_is_better:
        return False
    elif metric_lower in higher_is_better:
        return True

    # Default: if it's a regression metric not in the lists, assume lower is better
    # Otherwise assume higher is better
    if metric_lower in [m.lower() for m in RegrSupportedMetrics]:
        return False
    return True


def find_best_indices(numeric_values: pd.Series, higher_is_better: bool) -> list:
    """
    Find all indices with the best value (handles ties).

    Args:
        numeric_values: Series of numeric values
        higher_is_better: If True, find max values; if False, find min values

    Returns:
        List of indices with the best value
    """
    if numeric_values.notna().sum() == 0:
        return []

    if higher_is_better:
        best_value = numeric_values.max()
    else:
        best_value = numeric_values.min()

    # Find all indices with the best value (handles ties)
    best_indices = numeric_values[numeric_values == best_value].index.tolist()
    return best_indices


def prepare_dataframe_for_csv(
    df: pd.DataFrame, model_col: str = 'Model'
) -> pd.DataFrame:
    """Return a copy of ``df`` with human-friendly model names for CSV export."""

    csv_df = df.copy()
    if model_col in csv_df.columns:
        mapped = csv_df[model_col].map(MODEL_TO_COLUMN)
        csv_df[model_col] = mapped.fillna(csv_df[model_col])
    return csv_df


def save_to_both_locations(
    content: str | pd.DataFrame, relative_path: str, is_csv: bool = False
):
    """
    Save content to both the local results directory and the external output directory.

    Args:
        content: The content to write (str for text files, DataFrame for CSV files)
        relative_path: The relative path within the results directory
        is_csv: Whether this is a CSV file (handled differently)
    """
    for base_dir in [LOCAL_OUTPUT_DIR, OVERLEAF_OUTPUT_DIR]:
        # If Overleaf dir does not exist, log and skip saving there
        if base_dir == OVERLEAF_OUTPUT_DIR and not base_dir.exists():
            logger.warning(
                f'Overleaf output dir does not exist ({OVERLEAF_OUTPUT_DIR}), skipping save to Overleaf'
            )
            continue

        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if is_csv:
                # content expected to be a DataFrame
                content.to_csv(path, index=False)
            else:
                # content expected to be a string
                path.write_text(content)
            logger.info(f'Saved to: {path}')
        except Exception as e:
            logger.exception(f'Failed to save {relative_path} to {base_dir}: {e}')


def keep_only_all_eval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the AUROC CSV and drop unused columns.
    """
    df = df.drop(
        columns=[
            'Seen subject unseen item',
            'Unseen subject seen item',
            'Unseen subject unseen item',
        ],
        errors='ignore',
    )
    return df


def build_wide_df(
    df_discri_eval: pd.DataFrame,
    df_reg_eval: pd.DataFrame,
    include_regression: bool = True,
) -> pd.DataFrame:
    """
    Given a filtered DataFrame for one eval split, pivot it so that each
    row is a model (in MODEL_ORDER), each column is a task (in TASK_ORDER),
    and missing values are kept as empty strings.

    Args:
        df_discri_eval: DataFrame with classification metrics
        df_reg_eval: DataFrame with regression metrics
        include_regression: If True, include regression tasks. If False, only classification tasks.
    """
    # keep in df_reg_eval only data_tasks which are for regression
    if not df_reg_eval.empty:
        df_reg_eval = df_reg_eval[df_reg_eval['Data'].isin(REG_TASKS)].reset_index(
            drop=True
        )
        # replace some ML columns names in the regression DataFrame using ML_REGRESSION_TO_CLASSIFICATION
        for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
            df_reg_eval.loc[df_reg_eval['Model'] == ml_col, 'Model'] = dl_col

    # keep in df_discri_eval only data_tasks which are for discrete metrics
    if not df_discri_eval.empty:
        df_discri_eval = df_discri_eval[~df_discri_eval['Data'].isin(REG_TASKS)]

    # concat based on include_regression flag
    if include_regression:
        if not df_discri_eval.empty and not df_reg_eval.empty:
            df_eval = pd.concat([df_discri_eval, df_reg_eval], ignore_index=True)
        elif not df_discri_eval.empty:
            df_eval = df_discri_eval
        elif not df_reg_eval.empty:
            df_eval = df_reg_eval
        else:
            df_eval = pd.DataFrame()
    else:
        df_eval = df_discri_eval

    # start with the full list of models
    wide = pd.DataFrame({'Model': MODEL_ORDER_CLASSIFICATION})

    # flatten the groups into a single ordered list of datasets
    task_order = [ds for grp in GROUPS.values() for ds in grp]

    # Filter task_order based on include_regression flag
    if not include_regression:
        task_order = [ds for ds in task_order if ds not in REG_TASKS]

    if not df_eval.empty:
        for ds in task_order:
            col_name = DATASET_TO_COLUMN.get(ds, ds)
            subset = (
                df_eval[df_eval['Data'] == ds]
                .set_index('Model')['All']
                .rename(col_name)
            )
            # join will insert NaN for models with no data
            wide = wide.join(subset, on='Model')

    return wide.fillna('')


def build_task_wide_df_by_regime(
    all_metrics_data: dict,
    task: str,
    eval_type: str,
) -> pd.DataFrame:
    """
    Given all metrics data for a single eval split, build a wide DataFrame where:
    - Each row is a model (in MODEL_ORDER)
    - Columns are organized by regime (Unseen Reader, Unseen Text, Unseen Both, All)
    - For each regime, all relevant metrics are shown
    - Values are from the specified regime columns

    Args:
        all_metrics_data: Dict mapping metric name to (df_discri_eval, df_reg_eval) tuples
        task: The data task to extract metrics for
        eval_type: 'val' or 'test'

    Returns:
        Wide DataFrame with models as rows and (regime, metric) multi-index columns
    """
    # Start with the full list of models
    wide = pd.DataFrame({'Model': MODEL_ORDER_CLASSIFICATION})

    # Determine which metrics apply to this task
    is_regression_task = task in REG_TASKS

    # Process each metric
    for metric_name in all_metrics_data.keys():
        # Skip regression metric for classification tasks and vice versa
        if is_regression_task and metric_name not in RegrSupportedMetrics:
            continue
        if not is_regression_task and metric_name in RegrSupportedMetrics:
            continue

        # Get the appropriate dataframe
        df_discri_eval, df_reg_eval = all_metrics_data[metric_name]
        if metric_name in RegrSupportedMetrics:
            df_eval = df_reg_eval.copy()
            if not df_eval.empty:
                for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
                    df_eval.loc[df_eval['Model'] == ml_col, 'Model'] = dl_col
        else:
            df_eval = df_discri_eval

        # Extract data for this task
        if not df_eval.empty:
            task_data = df_eval[df_eval['Data'] == task]
            if not task_data.empty:
                # For each regime column, add it to the wide dataframe
                for source_col, regime_label in REGIME_COLS.items():
                    if source_col in task_data.columns:
                        col_name = f'{regime_label}_{METRICS_LABELS.get(metric_name, metric_name)}'
                        subset = task_data.set_index('Model')[source_col].rename(
                            col_name
                        )
                        wide = wide.join(subset, on='Model')

    return wide.fillna('')


def build_task_wide_df(
    all_metrics_data_val: dict,
    all_metrics_data_test: dict,
    task: str,
) -> pd.DataFrame:
    """
    Given all metrics data for val and test splits, build a wide DataFrame where:
    - Each row is a model (in MODEL_ORDER)
    - Each column pair is (metric_val, metric_test)
    - Values are from the 'All' column for the specified task

    Args:
        all_metrics_data_val: Dict mapping metric name to (df_discri_eval, df_reg_eval) tuples for validation
        all_metrics_data_test: Dict mapping metric name to (df_discri_eval, df_reg_eval) tuples for test
        task: The data task to extract metrics for

    Returns:
        Wide DataFrame with models as rows and metric columns for both val and test
    """
    # Start with the full list of models
    wide = pd.DataFrame({'Model': MODEL_ORDER_CLASSIFICATION})

    # Determine which metrics apply to this task
    is_regression_task = task in REG_TASKS

    # Process each metric for both val and test
    for metric_name in all_metrics_data_test.keys():
        # Skip regression metric for classification tasks and vice versa
        if is_regression_task and metric_name not in RegrSupportedMetrics:
            continue
        if not is_regression_task and metric_name in RegrSupportedMetrics:
            continue

        # Process validation data
        df_discri_eval_val, df_reg_eval_val = all_metrics_data_val[metric_name]
        if metric_name in RegrSupportedMetrics:
            df_eval_val = df_reg_eval_val.copy()
            if not df_eval_val.empty:
                for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
                    df_eval_val.loc[df_eval_val['Model'] == ml_col, 'Model'] = dl_col
        else:
            df_eval_val = df_discri_eval_val

        # Process test data
        df_discri_eval_test, df_reg_eval_test = all_metrics_data_test[metric_name]
        if metric_name in RegrSupportedMetrics:
            df_eval_test = df_reg_eval_test.copy()
            if not df_eval_test.empty:
                for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
                    df_eval_test.loc[df_eval_test['Model'] == ml_col, 'Model'] = dl_col
        else:
            df_eval_test = df_discri_eval_test

        # Extract validation data for this task
        if not df_eval_val.empty:
            task_data_val = df_eval_val[df_eval_val['Data'] == task]
            if not task_data_val.empty:
                col_name = f'{METRICS_LABELS.get(metric_name, metric_name)}_Val'
                subset = task_data_val.set_index('Model')['All'].rename(col_name)
                wide = wide.join(subset, on='Model')

        # Extract test data for this task
        if not df_eval_test.empty:
            task_data_test = df_eval_test[df_eval_test['Data'] == task]
            if not task_data_test.empty:
                col_name = f'{METRICS_LABELS.get(metric_name, metric_name)}_Test'
                subset = task_data_test.set_index('Model')['All'].rename(col_name)
                wide = wide.join(subset, on='Model')

    return wide.fillna('')


def generate_latex_table_per_task_by_regime(
    wide: pd.DataFrame,
    task: str,
    eval_type: str,
) -> str:
    """
    Produce a complete LaTeX table string for a specific task, broken down by regime,
    with all metrics shown for each regime.

    Args:
        wide: Wide DataFrame with model results (rows=models, cols=(regime, metric) combinations)
        task: The data task key (e.g., 'CopCo_RCS')
        eval_type: 'val' or 'test'
    """

    if wide.empty:
        return ''

    # Get all columns except 'Model'
    all_cols = [col for col in wide.columns if col != 'Model']

    if not all_cols:
        return ''

    # Organize columns by regime and metric
    # Column format is: {Regime}_{Metric}
    regimes = ['Unseen Reader', 'Unseen Text', 'Unseen Text \\& Reader', 'Average']

    # Extract unique metrics from columns
    metrics = []
    for col in all_cols:
        for regime in regimes:
            if col.startswith(f'{regime}_'):
                metric = col[len(regime) + 1 :]
                if metric not in metrics:
                    metrics.append(metric)

    # Build column format: one column for model name, then columns for each regime group
    # Each regime has one column per metric
    col_fmt_parts = ['l|']
    for regime in regimes:
        regime_metrics = [m for m in metrics if f'{regime}_{m}' in all_cols]
        if regime_metrics:
            col_fmt_parts.append('c' * len(regime_metrics))
            col_fmt_parts.append('|')
    col_fmt = ''.join(col_fmt_parts)

    # Get task display name
    task_label = DATASET_TO_COLUMN.get(task, task)

    # Determine eval type label
    eval_label = 'validation' if eval_type == 'val' else 'test'

    # Header: caption, resizebox, begin tabular
    hdr = dedent(f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Model performance on the {task_label} for the \\textbf{{{eval_label}}} set.}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}{col_fmt}@{{}}}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{Method}}}}""")

    # Add regime multicolumn headers
    for regime in regimes:
        regime_metrics = [m for m in metrics if f'{regime}_{m}' in all_cols]
        if regime_metrics:
            hdr += f' & \\multicolumn{{{len(regime_metrics)}}}{{c|}}{{\\textbf{{{regime}}}}}'
    hdr += ' \\\\\n'

    # Add cmidrules for each regime group
    cmid = []
    offset = 2
    for regime in regimes:
        regime_metrics = [m for m in metrics if f'{regime}_{m}' in all_cols]
        if regime_metrics:
            end = offset + len(regime_metrics) - 1
            cmid.append(f'\\cmidrule(lr){{{offset}-{end}}}')
            offset = end + 1
    hdr += ' '.join(cmid) + '\n'

    # Second header row: metric labels
    hdr += ' '
    for regime in regimes:
        regime_metrics = [m for m in metrics if f'{regime}_{m}' in all_cols]
        for metric in regime_metrics:
            hdr += f' & \\textbf{{{metric}}}'
    hdr += ' \\\\\n\\midrule\n'

    # Find best model for each (regime, metric) column
    best_by_col = {}
    for col in all_cols:
        # Extract numeric values for comparison
        numeric_values = wide[col].apply(extract_numeric_value)

        # Extract metric name from column (format: {Regime}_{Metric})
        for regime in regimes:
            if col.startswith(f'{regime}_'):
                metric_name = col[len(regime) + 1 :]
                break
        else:
            metric_name = col

        # Determine if higher is better for this metric
        higher_is_better = is_metric_higher_better(metric_name)

        # Find all best indices (handles ties)
        best_indices = find_best_indices(numeric_values, higher_is_better)
        if best_indices:
            best_by_col[col] = [wide.loc[idx, 'Model'] for idx in best_indices]

    # Body rows
    body = ''
    for _, row in wide.iterrows():
        first_cell = MODEL_TO_COLUMN[row['Model']]
        cells = [first_cell]

        for regime in regimes:
            regime_metrics = [m for m in metrics if f'{regime}_{m}' in all_cols]
            for metric in regime_metrics:
                col = f'{regime}_{metric}'
                value = str(row[col])

                # Format with subscript for std deviation
                value = format_value_with_subscript(value)

                # Bold if this is one of the best models for this regime+metric
                if (
                    row['Model'] in best_by_col.get(col, [])
                    and value != ''
                    and value != '-'
                ):
                    value = f'\\textbf{{{value}}}'

                cells.append(value)

        body += ' & '.join(cells)
        body += ' \\\\\n'
        if row['Model'] in ['Roberta', 'RandomForestMLArgs']:
            body += '\\midrule\n'

    # Table tail
    tail = dedent(rf"""
    \bottomrule
    \end{{tabular}}%
    }}
    \label{{tab:task-{task.lower()}-{eval_type}-regime}}
    \end{{table}}
    """)

    return hdr + body + tail


def generate_latex_table_per_task(
    wide: pd.DataFrame,
    task: str,
) -> str:
    """
    Produce a complete LaTeX table string for a specific task with metrics as columns,
    showing both validation and test results side-by-side.

    Args:
        wide: Wide DataFrame with model results (rows=models, cols=metrics with _Val and _Test suffixes)
        task: The data task key (e.g., 'CopCo_RCS')
    """

    if wide.empty:
        return ''

    # Get metric columns (all columns except 'Model')
    all_cols = [col for col in wide.columns if col != 'Model']

    if not all_cols:
        return ''

    # Group columns by metric (removing _Val and _Test suffixes)
    metric_names = []
    for col in all_cols:
        if col.endswith('_Val'):
            metric_name = col[:-4]
            if metric_name not in metric_names:
                metric_names.append(metric_name)
        elif col.endswith('_Test'):
            metric_name = col[:-5]
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    # Build column format: one column for model name, then 2 columns per metric (val, test)
    num_metric_cols = sum(
        1 for m in metric_names if f'{m}_Val' in all_cols or f'{m}_Test' in all_cols
    )
    col_fmt = 'l|' + 'cc|' * num_metric_cols

    # Get task display name
    task_label = DATASET_TO_COLUMN.get(task, task)

    # Header: caption, resizebox, begin tabular
    hdr = dedent(f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Model performance on \\textbf{{{task_label}}} task across different metrics for validation and test sets, averaged across folds. Best performing model per metric and split is shown in \\textbf{{bold}}.}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}{col_fmt}@{{}}}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{Method}}}}""")

    # Add metric multicolumn headers
    for metric_name in metric_names:
        has_val = f'{metric_name}_Val' in all_cols
        has_test = f'{metric_name}_Test' in all_cols
        if has_val or has_test:
            hdr += f' & \\multicolumn{{2}}{{c|}}{{\\textbf{{{metric_name}}}}}'
    hdr += ' \\\\\n'

    # Add cmidrules for each metric group
    cmid = []
    offset = 2
    for metric_name in metric_names:
        has_val = f'{metric_name}_Val' in all_cols
        has_test = f'{metric_name}_Test' in all_cols
        if has_val or has_test:
            end = offset + 1
            cmid.append(f'\\cmidrule(lr){{{offset}-{end}}}')
            offset = end + 1
    hdr += ' '.join(cmid) + '\n'

    # Second header row: Val/Test labels
    hdr += ' '
    for metric_name in metric_names:
        has_val = f'{metric_name}_Val' in all_cols
        has_test = f'{metric_name}_Test' in all_cols
        if has_val or has_test:
            hdr += ' & \\textbf{Val} & \\textbf{Test}'
    hdr += ' \\\\\n\\midrule\n'

    # Find best model for each metric column
    best_by_col = {}
    for col in all_cols:
        # Extract numeric values for comparison
        numeric_values = wide[col].apply(extract_numeric_value)

        # Extract metric name from column (format: {Metric}_Val or {Metric}_Test)
        if col.endswith('_Val') or col.endswith('_Test'):
            metric_name = col.rsplit('_', 1)[0]
        else:
            metric_name = col

        # Determine if higher is better for this metric
        higher_is_better = is_metric_higher_better(metric_name)

        # Find all best indices (handles ties)
        best_indices = find_best_indices(numeric_values, higher_is_better)
        if best_indices:
            best_by_col[col] = [wide.loc[idx, 'Model'] for idx in best_indices]

    # Body rows
    body = ''
    for _, row in wide.iterrows():
        first_cell = MODEL_TO_COLUMN[row['Model']]
        cells = [first_cell]

        for metric_name in metric_names:
            val_col = f'{metric_name}_Val'
            test_col = f'{metric_name}_Test'

            # Add validation value
            if val_col in all_cols:
                val_value = str(row[val_col])
                # Format with subscript for std deviation
                val_value = format_value_with_subscript(val_value)
                if (
                    row['Model'] in best_by_col.get(val_col, [])
                    and val_value != ''
                    and val_value != '-'
                ):
                    val_value = f'\\textbf{{{val_value}}}'
                cells.append(val_value)
            else:
                cells.append('')

            # Add test value
            if test_col in all_cols:
                test_value = str(row[test_col])
                # Format with subscript for std deviation
                test_value = format_value_with_subscript(test_value)
                if (
                    row['Model'] in best_by_col.get(test_col, [])
                    and test_value != ''
                    and test_value != '-'
                ):
                    test_value = f'\\textbf{{{test_value}}}'
                cells.append(test_value)
            else:
                cells.append('')

        body += ' & '.join(cells)
        body += ' \\\\\n'
        if row['Model'] in ['Roberta', 'RandomForestMLArgs']:
            body += '\\midrule\n'

    # Table tail
    tail = dedent(rf"""
    \bottomrule
    \end{{tabular}}%
    }}
    \label{{tab:task-{task.lower()}}}
    \end{{table}}
    """)

    return hdr + body + tail


def generate_latex_table(
    wide: pd.DataFrame,
    eval_type: str,
    discrete_metric: str,
    reg_metric: str = 'Not regression',
    include_regression: bool = True,
) -> str:
    """
    Produce a complete LaTeX table string for a given eval split
    ('test' or 'val') using the wide DataFrame.

    Args:
        wide: Wide DataFrame with model results
        eval_type: 'test' or 'val'
        discrete_metric: Name of the discrete metric (accuracy, auroc, etc.)
        reg_metric: Label for regression metric
        include_regression: If True, include regression tasks. If False, only classification.
    """

    # Get actual tasks that have data from the wide dataframe columns
    # (excluding the 'Model' column)
    available_columns = [col for col in wide.columns if col != 'Model']

    # Only include tasks that are actually in the dataframe
    task_order = []
    for ds in [ds for grp in GROUPS.values() for ds in grp]:
        col_name = DATASET_TO_COLUMN.get(ds, ds)
        if col_name in available_columns:
            # Check if column has any non-empty data
            has_data = (wide[col_name] != '').any()
            if has_data:
                task_order.append(ds)

    if not task_order:
        # No tasks to display
        return ''

    # build column-format, inserting '|' before each group block
    special_split_point = 'CopCo_TYP'
    col_fmt = ['l']
    col_fmt.append('|')  # vertical line between groups

    for grp_name, grp_tasks in GROUPS.items():
        # Only include tasks from this group that are in task_order
        tasks_in_group = [t for t in grp_tasks if t in task_order]

        for i, task in enumerate(tasks_in_group):
            if task == special_split_point:
                col_fmt.append(':')  # placeholder for dashed line logic
            col_fmt.append('c')

        if tasks_in_group:  # Only add separator if group has tasks
            col_fmt.append('|')  # vertical line between groups

    col_fmt = ''.join(col_fmt)

    # Determine metric direction for caption
    metric_name = METRICS_LABELS.get(discrete_metric, discrete_metric)

    # Determine if this is a regression-only or classification-only table
    has_classification_tasks = any(ds not in REG_TASKS for ds in task_order)
    has_regression_tasks = any(ds in REG_TASKS for ds in task_order)

    if has_regression_tasks and not has_classification_tasks:
        # Regression-only table
        metric_direction = f'Lower {reg_metric} values indicate better performance'
        task_desc = f'\\textbf{{{reg_metric}}} values are presented for all tasks'
    elif has_classification_tasks and not has_regression_tasks:
        # Classification-only table
        metric_direction = f'Higher {metric_name} values indicate better performance'
        task_desc = f'\\textbf{{{metric_name}}} values are presented for all tasks'

    # header: caption, resizebox, begin tabular
    hdr = dedent(f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Model performance across benchmark tasks grouped into Reader and Reader \\& Text categories. {task_desc}, averaged across folds. {metric_direction}. Best performing model per task is shown in \\textbf{{bold}}.}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}{col_fmt}@{{}}}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{Method}}}}""")

    # multicolumn headers
    for grp_name, ds_list in GROUPS.items():
        # Filter to only tasks in this group that are in task_order
        tasks_in_group = [t for t in ds_list if t in task_order]
        if tasks_in_group:
            hdr += f' & \\multicolumn{{{len(tasks_in_group)}}}{{c|}}{{\\textbf{{{grp_name}}}}}'
    hdr += ' \\\\\n'

    # cmidrules
    cmid = []
    offset = 2
    for ds_list in GROUPS.values():
        # Filter to only tasks in this group that are in task_order
        tasks_in_group = [t for t in ds_list if t in task_order]
        if tasks_in_group:
            end = offset + len(tasks_in_group) - 1
            cmid.append(f'\\cmidrule(lr){{{offset}-{end}}}')
            offset = end + 1
    hdr += ' '.join(cmid) + '\n'

    # second header row: task acronyms
    hdr += ' & ' + ' & '.join(
        f'\\textbf{{{DATASET_TO_COLUMN[ds]}}}' for ds in task_order
    )
    hdr += ' \\\\\n\\midrule\n'
    formatted_hline = '\\addlinespace[1ex]\n\\hline\n\\addlinespace[1ex]\n'
    hdr += formatted_hline

    # Find best model for each task
    best_by_task = {}
    for ds in task_order:
        col_name = DATASET_TO_COLUMN[ds]
        if col_name not in wide.columns:
            continue

        # Extract numeric values for comparison
        numeric_values = wide[col_name].apply(extract_numeric_value)

        # Determine metric for this task
        if ds in REG_TASKS:
            # For regression tasks, use reg_metric to determine direction
            higher_is_better = is_metric_higher_better(reg_metric)
        else:
            # For classification tasks, use discrete_metric to determine direction
            higher_is_better = is_metric_higher_better(discrete_metric)

        # Find all best indices (handles ties)
        best_indices = find_best_indices(numeric_values, higher_is_better)
        if best_indices:
            best_by_task[col_name] = [wide.loc[idx, 'Model'] for idx in best_indices]

    # body rows
    body = ''
    for _, row in wide.iterrows():
        first_cell = MODEL_TO_COLUMN[row['Model']]
        cells = [first_cell]

        for ds in task_order:
            col_name = DATASET_TO_COLUMN[ds]
            value = str(row[col_name])

            # Format with subscript for std deviation
            value = format_value_with_subscript(value)

            # Bold if this is one of the best models for this task
            if (
                row['Model'] in best_by_task.get(col_name, [])
                and value != ''
                and value != '-'
            ):
                value = f'\\textbf{{{value}}}'

            cells.append(value)

        body += ' & '.join(cells)
        body += ' \\\\\n'
        if row['Model'] in ['Roberta', 'RandomForestMLArgs']:
            body += formatted_hline

    # table tail
    tail = dedent(rf"""
    \bottomrule
    \end{{tabular}}%
    }}
    \small
    \label{{tab:task-results-{eval_type}-{discrete_metric}}}
    \end{{table}}
    """)

    return hdr + body + tail


def compute_aggregated_results_across_all_metrics(
    all_metrics_data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute aggregated results across ALL metrics and tasks.

    Args:
        all_metrics_data: Dict mapping metric name to (df_discri_eval, df_reg_eval) tuples

    Returns:
        tuple of (normalized_agg_df, ranking_agg_df)
    """

    results_list = []
    rankings_list = []

    task_order = [ds for grp in GROUPS.values() for ds in grp]

    logger.info(
        f'Computing aggregated results across {len(all_metrics_data)} metrics and {len(task_order)} tasks'
    )

    # Process each metric
    for metric_name, (df_discri_eval, df_reg_eval) in all_metrics_data.items():
        # Prepare data for this metric
        if not df_reg_eval.empty:
            df_reg_eval = df_reg_eval[df_reg_eval['Data'].isin(REG_TASKS)].reset_index(
                drop=True
            )
            # Replace some ML column names in the regression DataFrame
            for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
                df_reg_eval.loc[df_reg_eval['Model'] == ml_col, 'Model'] = dl_col

        if not df_discri_eval.empty:
            df_discri_eval = df_discri_eval[
                ~df_discri_eval['Data'].isin(REG_TASKS)
            ].reset_index(drop=True)

        # Combine all data for this metric
        if not df_discri_eval.empty and not df_reg_eval.empty:
            df_combined = pd.concat([df_discri_eval, df_reg_eval], ignore_index=True)
        elif not df_discri_eval.empty:
            df_combined = df_discri_eval
        elif not df_reg_eval.empty:
            df_combined = df_reg_eval
        else:
            continue  # Skip if both are empty

        # Process each task for this metric
        for task in task_order:
            task_data = df_combined[df_combined['Data'] == task].copy()
            if task_data.empty:
                continue

            scores = task_data[['Model', 'All']].copy()
            scores = scores.dropna()

            if scores.empty:
                continue

            # Extract numeric values
            scores['numeric_score'] = scores['All'].apply(extract_numeric_value)
            scores = scores.dropna(subset=['numeric_score'])

            if len(scores) < 2:
                continue

            # Determine if higher is better based on the metric, not the task type
            higher_is_better = is_metric_higher_better(metric_name)

            if higher_is_better:
                # For metrics where higher is better (e.g., AUROC, R2)
                scores['rank'] = scores['numeric_score'].rank(
                    method='min', ascending=False
                )
                max_score = scores['numeric_score'].max()
                min_score = scores['numeric_score'].min()
                if max_score != min_score:
                    scores['normalized'] = (scores['numeric_score'] - min_score) / (
                        max_score - min_score
                    )
                else:
                    scores['normalized'] = 0.5
            else:
                # For metrics where lower is better (e.g., RMSE, MAE)
                scores['rank'] = scores['numeric_score'].rank(
                    method='min', ascending=True
                )
                max_val = scores['numeric_score'].max()
                min_val = scores['numeric_score'].min()
                if max_val != min_val:
                    scores['normalized'] = 1 - (scores['numeric_score'] - min_val) / (
                        max_val - min_val
                    )
                else:
                    scores['normalized'] = 0.5

            # Add task and metric info
            scores['Task'] = task
            scores['Metric'] = metric_name
            results_list.append(scores[['Model', 'Task', 'Metric', 'normalized']])
            rankings_list.append(scores[['Model', 'Task', 'Metric', 'rank']])

    if not results_list:
        logger.warning('No results to aggregate across metrics')
        return pd.DataFrame(), pd.DataFrame()

    # Combine all results across all metrics and tasks
    all_normalized = pd.concat(results_list, ignore_index=True)
    all_rankings = pd.concat(rankings_list, ignore_index=True)

    logger.info(
        f'Aggregating results for {len(all_normalized["Model"].unique())} models across {len(all_normalized["Task"].unique())} tasks and {len(all_normalized["Metric"].unique())} metrics'
    )

    # Compute aggregated normalized scores (mean across tasks and metrics)
    normalized_agg = all_normalized.groupby('Model')['normalized'].mean().reset_index()
    normalized_agg.columns = ['Model', 'Avg_Normalized_Score']
    normalized_agg = normalized_agg.sort_values('Avg_Normalized_Score', ascending=False)

    # Compute aggregated rankings (mean rank across tasks and metrics)
    ranking_agg = all_rankings.groupby('Model')['rank'].mean().reset_index()
    ranking_agg.columns = ['Model', 'Avg_Rank']
    ranking_agg = ranking_agg.sort_values('Avg_Rank', ascending=True)

    logger.info(
        f'Generated aggregated results: {len(normalized_agg)} models in normalized scores, {len(ranking_agg)} models in rankings'
    )

    return normalized_agg, ranking_agg


def compute_aggregated_results(
    df_discri_eval: pd.DataFrame, df_reg_eval: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute two aggregated versions of the results:
    1. Normalized scores: Normalize the scores then take mean across metrics and tasks
    2. Ranking: For each metric and task, compute ranking of all models, then average rank

    Args:
        df_discri_eval: DataFrame with discrete metrics (accuracy, auroc, etc.)
        df_reg_eval: DataFrame with regression metrics (rmse)

    Returns:
        tuple of (normalized_agg_df, ranking_agg_df)
    """

    # Prepare data for aggregation
    df_reg_eval = df_reg_eval[df_reg_eval['Data'].isin(REG_TASKS)].reset_index(
        drop=True
    )
    df_discri_eval = df_discri_eval[
        ~df_discri_eval['Data'].isin(REG_TASKS)
    ].reset_index(drop=True)

    # Replace some ML column names in the regression DataFrame
    for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
        df_reg_eval.loc[df_reg_eval['Model'] == ml_col, 'Model'] = dl_col

    # Combine all data
    df_combined = pd.concat([df_discri_eval, df_reg_eval], ignore_index=True)

    # Get all unique model-task combinations
    results_list = []
    rankings_list = []

    # Process each task
    task_order = [ds for grp in GROUPS.values() for ds in grp]

    logger.info(f'Processing {len(task_order)} tasks for aggregation: {task_order}')

    for task in task_order:
        task_data = df_combined[df_combined['Data'] == task].copy()
        if task_data.empty:
            logger.warning(f'No data found for task: {task}')
            continue

        # Get the metric column (should be 'All')
        scores = task_data[['Model', 'All']].copy()
        scores = scores.dropna()

        if scores.empty:
            logger.warning(f'No valid scores found for task: {task}')
            continue

        # Extract numeric values from the string format
        scores['numeric_score'] = scores['All'].apply(extract_numeric_value)
        scores = scores.dropna(subset=['numeric_score'])

        if len(scores) < 2:  # Need at least 2 models for meaningful comparison
            logger.warning(f'Less than 2 valid scores for task: {task}')
            continue

        logger.info(f'Processing task {task} with {len(scores)} models')

        # For RMSE (regression), lower is better, so we invert for normalization
        if task in REG_TASKS:
            # For ranking: rank ascending (lower RMSE = better rank)
            scores['rank'] = scores['numeric_score'].rank(method='min', ascending=True)
            # For normalization: invert RMSE so higher normalized score = better
            max_rmse = scores['numeric_score'].max()
            min_rmse = scores['numeric_score'].min()
            if max_rmse != min_rmse:
                scores['normalized'] = 1 - (scores['numeric_score'] - min_rmse) / (
                    max_rmse - min_rmse
                )
            else:
                scores['normalized'] = 0.5  # All same, assign middle value
        else:
            # For other metrics (accuracy, auroc, etc.), higher is better
            scores['rank'] = scores['numeric_score'].rank(method='min', ascending=False)
            # Normalize to 0-1 range
            max_score = scores['numeric_score'].max()
            min_score = scores['numeric_score'].min()
            if max_score != min_score:
                scores['normalized'] = (scores['numeric_score'] - min_score) / (
                    max_score - min_score
                )
            else:
                scores['normalized'] = 0.5  # All same, assign middle value

        # Add task info
        scores['Task'] = task
        results_list.append(scores[['Model', 'Task', 'normalized']])
        rankings_list.append(scores[['Model', 'Task', 'rank']])

    if not results_list:
        logger.warning('No results to aggregate')
        # Return empty DataFrames if no data
        return pd.DataFrame(), pd.DataFrame()

    # Combine all results
    all_normalized = pd.concat(results_list, ignore_index=True)
    all_rankings = pd.concat(rankings_list, ignore_index=True)

    logger.info(
        f'Aggregating results for {len(all_normalized["Model"].unique())} models across {len(all_normalized["Task"].unique())} tasks'
    )

    # Compute aggregated normalized scores (mean across tasks)
    normalized_agg = all_normalized.groupby('Model')['normalized'].mean().reset_index()
    normalized_agg.columns = ['Model', 'Avg_Normalized_Score']
    normalized_agg = normalized_agg.sort_values('Avg_Normalized_Score', ascending=False)

    # Compute aggregated rankings (mean rank across tasks)
    ranking_agg = all_rankings.groupby('Model')['rank'].mean().reset_index()
    ranking_agg.columns = ['Model', 'Avg_Rank']
    ranking_agg = ranking_agg.sort_values('Avg_Rank', ascending=True)

    logger.info(
        f'Generated aggregated results: {len(normalized_agg)} models in normalized scores, {len(ranking_agg)} models in rankings'
    )

    return normalized_agg, ranking_agg


def generate_aggregated_latex_table(
    normalized_agg: pd.DataFrame, ranking_agg: pd.DataFrame, eval_type: str
) -> tuple[str, pd.DataFrame]:
    """
    Generate a LaTeX table showing both aggregated results with feature types.
    """
    # Define feature types for each model (hardcoded based on the table provided)

    # Merge the two aggregations
    merged = pd.merge(normalized_agg, ranking_agg, on='Model', how='outer')

    # Format model names
    merged['Model_Display'] = merged['Model'].map(MODEL_TO_COLUMN)

    # Round values for display
    merged['Avg_Normalized_Score'] = merged['Avg_Normalized_Score'].round(3)
    merged['Avg_Rank'] = merged['Avg_Rank'].round(2)

    # Sort by the same order as other tables (MODEL_ORDER_CLASSIFICATION)
    # Create a mapping from model name to its position in MODEL_ORDER_CLASSIFICATION
    model_order_map = {model: i for i, model in enumerate(MODEL_ORDER_CLASSIFICATION)}
    merged['order'] = merged['Model'].map(model_order_map)
    merged = merged.sort_values('order')
    merged = merged.drop(columns=['order'])

    # Find best models (handles ties)
    best_norm_score_indices = find_best_indices(
        merged['Avg_Normalized_Score'], higher_is_better=True
    )
    best_rank_indices = find_best_indices(merged['Avg_Rank'], higher_is_better=False)

    # Build DataFrame representing the LaTeX table contents
    feature_cols = [
        'Layout',
        'Saccade/Fixation',
        'Word-Level',
        'Trial-Level',
        'Linguistic',
        'Embeddings',
    ]
    csv_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        model_key = row['Model']
        display_name = (
            row['Model_Display'] if pd.notna(row['Model_Display']) else model_key
        )
        features = FEATURE_TYPES.get(model_key, {})
        csv_row = {'Model': display_name}
        for feature_col in feature_cols:
            csv_row[feature_col] = features.get(feature_col, '-')
        csv_row['Avg Normalized Score'] = row['Avg_Normalized_Score']
        csv_row['Mean Rank'] = row['Avg_Rank']
        csv_rows.append(csv_row)

    csv_table = pd.DataFrame(csv_rows)

    header = dedent("""
    \\begin{table}[ht]
    \\centering
    \\caption{Feature types used by each model, and aggregated model performance across all benchmark tasks and metrics. \\textbf{Layout} stands for information about the position of the text or fixations on the screen. Eye movement features are divided into three levels of granularity: \\textbf{Saccades/Fixations} (e.g., fixation duration), \\textbf{Words} (e.g., total fixation duration on a given word), and \\textbf{Trial} (e.g., average total fixation duration across all the words during the trial). Text features are divided into: \\textbf{Linguistic} word properties (e.g., word frequency) and contextual word \\textbf{Embeddings} (e.g., RoBERTa embeddings). \\textbf{Average Normalized Score} is the mean of min-max normalized scores across all tasks and metrics (higher is better). \\textbf{Mean Rank} is the mean ranking across all tasks and metrics (lower is better). Best performing model for each aggregation metric is shown in \\textbf{bold}.}
    \\resizebox{\\linewidth}{!}{%
    \\begin{tabular}{l|c|ccc|cc||cc}
    \\toprule
    \\textbf{Model} & \\multicolumn{1}{c|}{\\textbf{Layout}} & \\multicolumn{3}{c|}{\\textbf{Eye movement features}} & \\multicolumn{2}{c||}{\\textbf{Text features}}  & \\multicolumn{2}{c}{\\textbf{Aggregated performance}}\\\\
    &  & \\makecell{Saccade/\\\\Fixation Level} & \\makecell{Word\\\\Level} & \\makecell{Trial\\\\Level} & Linguistic & Embeddings & Avg. Normalized Score$\\uparrow$ & Mean Rank$\\downarrow$\\\\
    \\midrule
    """)

    body = ''
    for idx, row in merged.iterrows():
        model_name = (
            row['Model_Display'] if pd.notna(row['Model_Display']) else row['Model']
        )

        # Get feature types for this model
        features = FEATURE_TYPES[row['Model']]

        # Format scores with bold for best
        if pd.notna(row['Avg_Normalized_Score']):
            norm_score = f'{row["Avg_Normalized_Score"]:.3f}'
            if idx in best_norm_score_indices:
                norm_score = f'\\textbf{{{norm_score}}}'
        else:
            norm_score = '--'

        if pd.notna(row['Avg_Rank']):
            avg_rank = f'{row["Avg_Rank"]:.2f}'
            if idx in best_rank_indices:
                avg_rank = f'\\textbf{{{avg_rank}}}'
        else:
            avg_rank = '--'

        # Build row with feature types and aggregated performance
        body += f'{model_name} & {features["Layout"]} & {features["Saccade/Fixation"]} & {features["Word-Level"]} & {features["Trial-Level"]} & {features["Linguistic"]} & {features["Embeddings"]} & {norm_score} & {avg_rank} \\\\\n'

        # Add horizontal lines after certain model groups
        if row['Model'] in ['Roberta', 'RandomForestMLArgs']:
            body += '\\midrule\n'

    footer = dedent(f"""
    \\bottomrule
    \\end{{tabular}}%
    }}
    \\label{{tab:features-per-model-results-{eval_type}}}
    \\end{{table}}
    """)

    return header + body + footer, csv_table.reset_index(drop=True)


def generate_breakdown_tables(df: pd.DataFrame, metric: str, metric_type: str):
    """
    For each task, generate a LaTeX table showing per-model performance
    across evaluation regimes: unseen subject, unseen item, unseen both, and all.
    """
    eval_cols = [
        'Unseen subject seen item',
        'Seen subject unseen item',
        'Unseen subject unseen item',
        'All',
    ]

    for task_key, task_label in DATASET_TO_COLUMN.items():
        df_task = df[df['Data'] == task_key].copy()

        # Ensure all models from MODEL_ORDER are present
        # Create a dataframe with all models
        all_models_df = pd.DataFrame({'Model': MODEL_ORDER_BY_METRIC_TYPE[metric_type]})

        # Merge with existing data, keeping all models
        if not df_task.empty:
            df_task = df_task[['Model'] + eval_cols]
            df_task = all_models_df.merge(df_task, on='Model', how='left')
        else:
            df_task = all_models_df
            for col in eval_cols:
                df_task[col] = ''

        # Replace NaN with '-'
        df_task[eval_cols] = df_task[eval_cols].fillna('-')

        # Replace empty strings with '-'
        for col in eval_cols:
            df_task[col] = df_task[col].replace('', '-')

        # Save original model names before formatting
        df_task['Model_Original'] = df_task['Model']

        # Format model names
        df_task['Model'] = df_task['Model'].map(MODEL_TO_COLUMN)
        df_task['Model'] = df_task['Model'].fillna(df_task['Model_Original'])

        # Find best model for each evaluation column
        # Determine if higher or lower is better based on metric
        is_higher_better = is_metric_higher_better(metric)

        best_indices = {}
        for col in eval_cols:
            numeric_vals = df_task[col].apply(extract_numeric_value)
            if numeric_vals.notna().any():
                # Find all best indices (handles ties)
                best_idx_list = find_best_indices(numeric_vals, is_higher_better)
                best_indices[col] = best_idx_list
            else:
                best_indices[col] = []

        # Build LaTeX
        header = dedent(
            rf"""
        \begin{{table}}[ht]
        \centering
        \caption{{{METRICS_LABELS[metric]} performance for task: \textbf{{{task_label}}}, broken down by evaluation regime.}}
        \resizebox{{\textwidth}}{{!}}{{%
        \begin{{tabular}}{{l|ccc|c}}
        \toprule
        \textbf{{Model}} & \textbf{{Unseen Reader}} & \textbf{{Unseen Text}} & \textbf{{Unseen Reader \& Text}} & \textbf{{All}} \\
        \midrule
        """
        )

        rows = ''
        for idx, row in df_task.iterrows():
            cells = [str(row['Model'])]

            # Format each evaluation column, bolding if it's the best
            for col in eval_cols:
                val = str(row[col])
                # Format with subscript for std deviation
                val = format_value_with_subscript(val)
                if idx in best_indices[col] and val != '-':
                    val = f'\\textbf{{{val}}}'
                cells.append(val)

            rows += ' & '.join(cells) + ' \\\\\n'
            if row['Model_Original'] in ['Roberta', 'RandomForestMLArgs']:
                rows += '\\midrule\n'

        footer = dedent(f"""
        \\bottomrule
        \\end{{tabular}}%
        }}
        \\label{{tab:task-breakdown-{task_key.lower()}-{metric}}}
        \\end{{table}}
        """)

        # Save to both locations
        latex_content = header + rows + footer
        save_to_both_locations(
            latex_content, f'breakdown/{task_key}_{metric}.tex', is_csv=False
        )
        csv_output = df_task.drop(columns=['Model_Original'], errors='ignore')
        save_to_both_locations(
            csv_output, f'breakdown/{task_key}_{metric}.csv', is_csv=True
        )


def generate_combined_table(
    df_auroc_test: pd.DataFrame,
    df_rmse_test: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """
    Generate a combined LaTeX table showing AUROC for classification tasks
    and RMSE for regression tasks in a single table.

    The table groups tasks by type:
    - Classification tasks (Reading Comprehension, Domain Expertise, Claim Verification, Dyslexia Detection)
    - Regression tasks (Subj. Text Difficulty, Reading Compr. Skill, Vocab. Knowledge)

    Args:
        df_auroc_test: DataFrame with AUROC results for test split
        df_rmse_test: DataFrame with RMSE results for test split

    Returns:
        Tuple containing the LaTeX table string and a DataFrame representation
    """

    all_tasks = classification_tasks + regression_tasks

    # Prepare RMSE data: replace ML model names with DL equivalents
    df_rmse_processed = df_rmse_test.copy()
    for ml_col, dl_col in ML_REGRESSION_TO_CLASSIFICATION.items():
        df_rmse_processed.loc[df_rmse_processed['Model'] == ml_col, 'Model'] = dl_col

    # Build wide dataframe with all models
    wide = pd.DataFrame({'Model': MODEL_ORDER_CLASSIFICATION})

    # Add AUROC values for classification tasks
    for task in classification_tasks:
        task_data = df_auroc_test[df_auroc_test['Data'] == task]
        if not task_data.empty:
            subset = task_data.set_index('Model')['All'].rename(task)
            wide = wide.join(subset, on='Model')

    # Add RMSE values for regression tasks
    for task in regression_tasks:
        task_data = df_rmse_processed[df_rmse_processed['Data'] == task]
        if not task_data.empty:
            subset = task_data.set_index('Model')['All'].rename(task)
            wide = wide.join(subset, on='Model')

    # Replace NaN with empty string
    wide = wide.fillna('')

    # Find best model for each task
    best_by_task = {}
    for task in all_tasks:
        if task not in wide.columns:
            continue

        numeric_values = wide[task].apply(extract_numeric_value)

        # For regression tasks, use RMSE (lower is better); for classification, use AUROC (higher is better)
        if task in regression_tasks:
            higher_is_better = is_metric_higher_better('rmse')
        else:
            higher_is_better = is_metric_higher_better('auroc')

        # Find all best indices (handles ties)
        best_indices = find_best_indices(numeric_values, higher_is_better)
        if best_indices:
            best_by_task[task] = [wide.loc[idx, 'Model'] for idx in best_indices]

    # Build table header
    num_classification = len(classification_tasks)
    num_regression = len(regression_tasks)

    col_fmt = 'l|' + 'c' * num_classification + '|' + 'c' * num_regression

    # Build second header row with task types
    header_row2 = ' '

    # Mapping of special commands per task
    special_cmds_before = {
        'OneStop_RC': r'',
        'SBSAT_RC': r'\newsetup{}',
        'PoTeC_RC': r'\newthing{}',
        'PoTeC_DE': r'\newthing{}',
        'IITBHGC_CV': r'\newthing{}',
        'CopCo_TYP': r'',
        'SB-SAT_REGR': r'',
        'CopCo_REGR': r'',
        'MECO_L2_REGR': r'\newthing{}',
    }
    special_cmds_after = {
        'OneStop_RC': r'',
        'SBSAT_RC': r'',
        'PoTeC_RC': r'',
        'PoTeC_DE': r'\woman{}+\page{}',
        'IITBHGC_CV': r'\woman{}+\page{}',
        'CopCo_TYP': r'\woman{}',
        'SB-SAT_REGR': r'\woman{}+\page{}',
        'CopCo_REGR': r'\woman{}',
        'MECO_L2_REGR': r'\woman{}',
    }

    # Reading Comprehension tasks (first 3 classification tasks)
    reading_compr_tasks = ['OneStop_RC', 'SBSAT_RC', 'PoTeC_RC']
    num_reading_compr = len(
        [t for t in reading_compr_tasks if t in classification_tasks]
    )
    if num_reading_compr > 0:
        header_row2 += f' & \\multicolumn{{{num_reading_compr}}}{{c}}{{Reading Comprehension\\woman{{}}+\\page{{}}}}'

    # Other classification tasks
    other_class_tasks = ['PoTeC_DE', 'IITBHGC_CV', 'CopCo_TYP']
    for task in other_class_tasks:
        if task in classification_tasks:
            label, _ = task_headers.get(task, ('Unknown', 'Unknown'))
            cmd = special_cmds_before.get(task, '')
            cmd2 = special_cmds_after.get(task, '')
            if cmd:
                label = f'{cmd} {label} {cmd2}'
            header_row2 += f' & \\makecell{{{label}}}'

    # Regression tasks
    for task in regression_tasks:
        label, _ = task_headers.get(task, ('Unknown', 'Unknown'))
        cmd = special_cmds_before.get(task, '')
        cmd2 = special_cmds_after.get(task, '')
        if cmd:
            label = f'{cmd} {label} {cmd2}'
        header_row2 += f' & \\makecell{{{label}}}'

    header_row2 += '\\\\\n'

    # Build third header row with dataset names
    header_row3 = ' '
    for task in classification_tasks + regression_tasks:
        _, dataset = task_headers.get(task, ('Unknown', 'Unknown'))
        header_row3 += f' & {dataset}'
    header_row3 += '\\\\\n'

    header = dedent(f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{Model performance across the benchmark tasks and datasets on test data. \\textbf{{AUROC}} (higher is better) for classification tasks and \\textbf{{RMSE}} (lower is better) for regression tasks. The best performing model per task and dataset is shown in \\textbf{{bold}}. Reported values indicate mean~$\\pm$~standard error across folds. The tasks belong to two categories, where \\woman{{}} indicates a reader characteristic prediction task and \\woman{{}}+\\page{{}} an interaction of a reader with a text. \\newthing{{}} indicates new tasks and task-dataset combinations introduced in \\benchmarkname. \\newsetup{{}} indicates a new experimental setup for the task-dataset combination.}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}{col_fmt}@{{}}}}
    \\toprule
     & \\multicolumn{{{num_classification}}}{{c|}}{{\\textbf{{Classification (AUROC$\\uparrow$)}}}} & \\multicolumn{{{num_regression}}}{{c}}{{\\textbf{{Regression (RMSE$\\downarrow$)}}}} \\\\
    \\addlinespace{header_row2}{header_row3}\\midrule
    \\addlinespace
    """)

    # Build table body
    body = ''
    models_in_table: list[str] = []
    for _, row in wide.iterrows():
        model_name = str(row['Model'])

        # Skip models with no data - check if all task values are empty
        has_data = False
        for task in all_tasks:
            if task in wide.columns:
                val_str = str(row[task])
                if val_str and val_str != '' and val_str != 'nan':
                    has_data = True
                    break

        if not has_data:
            continue

        models_in_table.append(model_name)

        # First column: model name
        model_label = MODEL_TO_COLUMN.get(model_name, model_name)
        if model_label is None:
            model_label = model_name
        cells: list[str] = [model_label]

        # Add values for each task (only tasks with data)
        for task in classification_tasks + regression_tasks:
            if task not in wide.columns:
                cells.append('-')
            else:
                value = str(row[task])
                if value == '' or value == 'nan':
                    cells.append('-')
                else:
                    # Format with subscript for std deviation
                    if ' ± ' in value:
                        mean, std = value.split(' ± ')
                        formatted_value = f'{mean}\\textsubscript{{±{std}}}'
                    else:
                        formatted_value = value

                    # Bold if one of the best models for this task
                    if model_name in best_by_task.get(task, []):
                        formatted_value = f'\\textbf{{{formatted_value}}}'

                    cells.append(formatted_value)

        body += ' & '.join(cells) + '\\\\\n'

        # Add horizontal lines after certain model groups
        if model_name in ['Roberta', 'RandomForestMLArgs']:
            body += '\\addlinespace[1ex]\n\\hline\n\\addlinespace[1ex]\n'

    # Build table footer
    footer = dedent("""
    \\bottomrule
    \\end{tabular}%
    }
    \\small
    \\label{tab:task-results-combined}
    \\end{table}
    """)

    latex_table = header + body + footer

    wide_filtered = wide[wide['Model'].isin(models_in_table)].reset_index(drop=True)
    csv_df = prepare_dataframe_for_csv(wide_filtered)

    return latex_table, csv_df


def main():
    # Create both output directories
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Get all tasks
    task_order = [ds for grp in GROUPS.values() for ds in grp]

    # Collect all metrics data for both val and test splits
    logger.info('Loading all metrics data for validation and test splits')

    # We need two sets of data:
    # 1. Filtered (only 'All' regime) for backward compatibility tables
    # 2. Full (all regimes) for new regime breakdown tables
    all_metrics_data_val_filtered = {}
    all_metrics_data_test_filtered = {}
    all_metrics_data_val_full = {}
    all_metrics_data_test_full = {}

    # Load regression metrics
    for reg_metric in RegrSupportedMetrics:
        reg_df = pd.read_csv(CSV_BASE_PATH / f'{reg_metric.value}.csv')
        filtered_reg_df = keep_only_all_eval(reg_df)

        # Split by eval type for filtered data (only 'All' regime)
        df_reg_eval_val_filtered = filtered_reg_df[
            filtered_reg_df['Eval Type'] == 'val'
        ]
        df_reg_eval_test_filtered = filtered_reg_df[
            filtered_reg_df['Eval Type'] == 'test'
        ]

        all_metrics_data_val_filtered[reg_metric.value] = (
            pd.DataFrame(),
            df_reg_eval_val_filtered,
        )
        all_metrics_data_test_filtered[reg_metric.value] = (
            pd.DataFrame(),
            df_reg_eval_test_filtered,
        )

        # Split by eval type for full data (all regimes)
        df_reg_eval_val_full = reg_df[reg_df['Eval Type'] == 'val']
        df_reg_eval_test_full = reg_df[reg_df['Eval Type'] == 'test']

        all_metrics_data_val_full[reg_metric.value] = (
            pd.DataFrame(),
            df_reg_eval_val_full,
        )
        all_metrics_data_test_full[reg_metric.value] = (
            pd.DataFrame(),
            df_reg_eval_test_full,
        )

    # Load classification metrics
    for discri_metric in DiscriSupportedMetrics:
        discri_df = pd.read_csv(CSV_BASE_PATH / f'{discri_metric.value}.csv')
        filtered_discri_df = keep_only_all_eval(discri_df)

        # Split by eval type for filtered data (only 'All' regime)
        df_discri_eval_val_filtered = filtered_discri_df[
            filtered_discri_df['Eval Type'] == 'val'
        ]
        df_discri_eval_test_filtered = filtered_discri_df[
            filtered_discri_df['Eval Type'] == 'test'
        ]

        all_metrics_data_val_filtered[discri_metric.value] = (
            df_discri_eval_val_filtered,
            pd.DataFrame(),
        )
        all_metrics_data_test_filtered[discri_metric.value] = (
            df_discri_eval_test_filtered,
            pd.DataFrame(),
        )

        # Split by eval type for full data (all regimes)
        df_discri_eval_val_full = discri_df[discri_df['Eval Type'] == 'val']
        df_discri_eval_test_full = discri_df[discri_df['Eval Type'] == 'test']

        all_metrics_data_val_full[discri_metric.value] = (
            df_discri_eval_val_full,
            pd.DataFrame(),
        )
        all_metrics_data_test_full[discri_metric.value] = (
            df_discri_eval_test_full,
            pd.DataFrame(),
        )

    # NEW: Generate per-task tables with regime breakdown, separate for val and test
    logger.info(
        'Generating per-task tables with regime breakdown for validation and test splits'
    )
    for task in task_order:
        task_label = DATASET_TO_COLUMN.get(task, task)
        logger.info(f'  Processing task: {task} ({task_label})')

        # Generate validation table with regime breakdown
        wide_task_val = build_task_wide_df_by_regime(
            all_metrics_data_val_full, task, 'val'
        )
        if not wide_task_val.empty:
            csv_val = prepare_dataframe_for_csv(wide_task_val)
            non_model_cols = [c for c in csv_val.columns if c != 'Model']
            if non_model_cols:
                csv_val = csv_val[(csv_val[non_model_cols] != '').any(axis=1)]
            csv_val = csv_val.reset_index(drop=True)
            save_to_both_locations(csv_val, f'{task}_val.csv', is_csv=True)
            latex_table_val = generate_latex_table_per_task_by_regime(
                wide_task_val, task, 'val'
            )
            if latex_table_val:
                tex_filename = f'{task}_val.tex'
                save_to_both_locations(latex_table_val, tex_filename, is_csv=False)
                logger.info(f'    Saved validation regime table: {tex_filename}')
        else:
            logger.warning(f'    No validation data for task: {task}')

        # Generate test table with regime breakdown
        wide_task_test = build_task_wide_df_by_regime(
            all_metrics_data_test_full, task, 'test'
        )
        if not wide_task_test.empty:
            csv_test = prepare_dataframe_for_csv(wide_task_test)
            non_model_cols = [c for c in csv_test.columns if c != 'Model']
            if non_model_cols:
                csv_test = csv_test[(csv_test[non_model_cols] != '').any(axis=1)]
            csv_test = csv_test.reset_index(drop=True)
            save_to_both_locations(csv_test, f'{task}_test.csv', is_csv=True)
            latex_table_test = generate_latex_table_per_task_by_regime(
                wide_task_test, task, 'test'
            )
            if latex_table_test:
                tex_filename = f'{task}_test.tex'
                save_to_both_locations(latex_table_test, tex_filename, is_csv=False)
                logger.info(f'    Saved test regime table: {tex_filename}')
        else:
            logger.warning(f'    No test data for task: {task}')

    # KEEP OLD: Generate per-task tables with metrics as columns, showing both val and test
    if do_metric_level_tables:
        logger.info(
            'Generating per-task tables with validation and test side-by-side (backward compatibility)'
        )
        for task in task_order:
            task_label = DATASET_TO_COLUMN.get(task, task)
            logger.info(f'  Processing task: {task} ({task_label})')

            # Build wide dataframe for this task with both val and test
            wide_task = build_task_wide_df(
                all_metrics_data_val_filtered, all_metrics_data_test_filtered, task
            )

            if not wide_task.empty:
                csv_wide_task = prepare_dataframe_for_csv(wide_task)
                non_model_cols = [c for c in csv_wide_task.columns if c != 'Model']
                if non_model_cols:
                    csv_wide_task = csv_wide_task[
                        (csv_wide_task[non_model_cols] != '').any(axis=1)
                    ]
                csv_wide_task = csv_wide_task.reset_index(drop=True)
                save_to_both_locations(
                    csv_wide_task, f'results_table_{task}.csv', is_csv=True
                )
                # Generate LaTeX table
                latex_table = generate_latex_table_per_task(wide_task, task)
                if latex_table:
                    tex_filename = f'results_table_{task}.tex'
                    save_to_both_locations(latex_table, tex_filename, is_csv=False)
                    logger.info(f'    Saved LaTeX: {tex_filename}')
            else:
                logger.warning(f'    No data for task: {task}')

    # Generate combined table with AUROC and RMSE for test split
    logger.info('Generating combined table with AUROC and RMSE for test split')
    auroc_df_test = pd.read_csv(CSV_BASE_PATH / 'auroc.csv')
    auroc_df_test = keep_only_all_eval(auroc_df_test)
    auroc_df_test = auroc_df_test[auroc_df_test['Eval Type'] == 'test']
    rmse_df_test = pd.read_csv(CSV_BASE_PATH / 'rmse.csv')
    rmse_df_test = keep_only_all_eval(rmse_df_test)
    rmse_df_test = rmse_df_test[rmse_df_test['Eval Type'] == 'test']

    combined_latex, combined_df = generate_combined_table(auroc_df_test, rmse_df_test)
    save_to_both_locations(combined_latex, 'results_tasks_combined.tex', is_csv=False)
    save_to_both_locations(combined_df, 'results_tasks_combined.csv', is_csv=True)
    logger.info('  Saved combined table: results_tasks_combined.tex')

    # Also generate the original metric-based tables for backward compatibility
    logger.info('Generating metric-based tables for test splits')

    for split in ['test']:
        logger.info(f'  Processing {split} split')

        all_metrics_data = (
            all_metrics_data_test_filtered
            if split == 'test'
            else all_metrics_data_val_filtered
        )

        if do_metric_level_tables:
            for reg_metric in RegrSupportedMetrics:
                df_reg_eval = all_metrics_data[reg_metric.value][1]

                # Generate REGRESSION-ONLY table (RMSE with only regression tasks)
                wide_regression = build_wide_df(
                    pd.DataFrame(),  # Empty classification df
                    df_reg_eval,
                    include_regression=True,
                )
                if not wide_regression.empty:
                    csv_regression = prepare_dataframe_for_csv(wide_regression)
                    non_model_cols = [c for c in csv_regression.columns if c != 'Model']
                    if non_model_cols:
                        csv_regression = csv_regression[
                            (csv_regression[non_model_cols] != '').any(axis=1)
                        ]
                    csv_regression = csv_regression.reset_index(drop=True)
                    save_to_both_locations(
                        csv_regression,
                        f'results_table_{split}_{reg_metric.value}.csv',
                        is_csv=True,
                    )
                latex_regression = generate_latex_table(
                    wide_regression,
                    eval_type=split,
                    discrete_metric=reg_metric.value,
                    reg_metric=METRICS_LABELS[reg_metric.value],
                    include_regression=True,
                )
                save_to_both_locations(
                    latex_regression,
                    f'results_table_{split}_{reg_metric.value}.tex',
                    is_csv=False,
                )

            # Process each classification metric
            for discri_metric in DiscriSupportedMetrics:
                df_discri_eval = all_metrics_data[discri_metric.value][0]

                # Generate CLASSIFICATION-ONLY table (only classification tasks)
                wide_classification = build_wide_df(
                    df_discri_eval,
                    pd.DataFrame(),  # Empty regression df
                    include_regression=False,
                )
                if not wide_classification.empty:
                    csv_classification = prepare_dataframe_for_csv(wide_classification)
                    non_model_cols = [
                        c for c in csv_classification.columns if c != 'Model'
                    ]
                    if non_model_cols:
                        csv_classification = csv_classification[
                            (csv_classification[non_model_cols] != '').any(axis=1)
                        ]
                    csv_classification = csv_classification.reset_index(drop=True)
                    save_to_both_locations(
                        csv_classification,
                        f'results_table_{split}_{discri_metric.value}.csv',
                        is_csv=True,
                    )
                latex_classification = generate_latex_table(
                    wide_classification,
                    eval_type=split,
                    discrete_metric=discri_metric.value,
                    reg_metric='Not regression',
                    include_regression=False,
                )
                save_to_both_locations(
                    latex_classification,
                    f'results_table_{split}_{discri_metric.value}.tex',
                    is_csv=False,
                )

        # Generate AGGREGATED results across ALL metrics
        logger.info(
            f'Computing aggregated results across all metrics for {split} split'
        )
        normalized_agg, ranking_agg = compute_aggregated_results_across_all_metrics(
            all_metrics_data
        )

        if not normalized_agg.empty and not ranking_agg.empty:
            # Generate aggregated LaTeX table
            aggregated_latex, aggregated_table_df = generate_aggregated_latex_table(
                normalized_agg, ranking_agg, f'{split}_all_metrics'
            )
            save_to_both_locations(
                aggregated_latex,
                f'aggregated_results_{split}_all_metrics.tex',
                is_csv=False,
            )
            save_to_both_locations(
                aggregated_table_df,
                f'aggregated_results_{split}_all_metrics.csv',
                is_csv=True,
            )

    logger.info('All tables generated successfully!')


if __name__ == '__main__':
    main()
