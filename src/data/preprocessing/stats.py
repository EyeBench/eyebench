from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.configs.constants import STATS_FOLDER, DataSets, DataType
from src.configs.data import DataArgs, get_data_args
from src.data.utils import load_raw_data

logger.add('logs/preprocessing.log', level='INFO')


def summarize_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    summary = []
    # TODO round digits properly
    for col in tqdm(df.columns, desc='Processing columns'):
        try:
            nunique = df[col].nunique()
        except TypeError:
            logger.warning(f'Skipping non parsable column {col}')
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        ):
            stats = {
                'column': col,
                'type': 'numeric',
                'count': df[col].count(),
                'percent_null': round((df[col].isna().sum() / len(df[col])) * 100, 1),
                'min': df[col].min(),
                'quantile_10': df[col].quantile(0.1),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'quantile_90': df[col].quantile(0.9),
                'max': df[col].max(),
                'std': df[col].std(),
                'nunique': nunique,
                'distribution': None,
            }
        elif pd.api.types.is_bool_dtype(df[col]):
            stats = {
                'column': col,
                'type': 'boolean',
                'count': df[col].count(),
                'percent_null': round((df[col].isna().sum() / len(df[col])) * 100, 1),
                'min': None,
                'quantile_10': None,
                'mean': None,
                'median': None,
                'quantile_90': None,
                'max': None,
                'std': None,
                'nunique': nunique,
                'distribution': None,
            }
        else:
            stats = {
                'column': col,
                'type': 'categorical',
                'count': df[col].count(),
                'percent_null': round((df[col].isna().sum() / len(df[col])) * 100, 1),
                'min': None,
                'quantile_10': None,
                'mean': None,
                'median': None,
                'quantile_90': None,
                'max': None,
                'std': None,
                'nunique': nunique,
                'distribution': None,
            }
        if nunique <= 15:
            value_counts = (df[col].value_counts(normalize=True) * 100).round(1)
            stats['distribution'] = value_counts.to_dict()

        summary.append(stats)

    return pd.DataFrame(summary).sort_values(by=['column'], ascending=True)


def format_stats(series, rounding=0):
    return f'{series.mean():.{rounding + 1}f}±{series.std():.{rounding}f} ({series.min():.{rounding}f}-{series.max():.{rounding}f})'


def compute_dataset_profile(
    dataset_name: str,
    mode: DataType,
    curr_df: pd.DataFrame,
    output_path: Path,
    data_args: DataArgs,
):
    """
    Computes and updates per-dataset profile stats in a shared CSV.
    """
    logger.info(f'---- Stats for: {dataset_name} ({mode}) ----')
    stats = {}
    item_col = data_args.unique_item_column
    participant_col = data_args.subject_column
    task_and_labels = data_args.tasks
    additional_info = {
        key: getattr(data_args, key, None)
        for key in ['text_source', 'text_language', 'text_domain', 'text_type']
        if hasattr(data_args, key)
    }
    if mode == DataType.IA:
        stats = {
            'n_participants': curr_df[participant_col].nunique(),
            'n_items': curr_df[item_col].nunique(),
            'n_words': curr_df.shape[0],
            'n_trials': curr_df['unique_trial_id'].nunique(),
            'n_words_per_participant': format_stats(
                curr_df.groupby(participant_col).size()
            ),
            'n_words_per_item': format_stats(curr_df.groupby(item_col).size()),
        }

        trial_level = curr_df.drop_duplicates(subset=['unique_trial_id'])
        # label stats

        if dataset_name == 'OneStop':
            stats['n_words_corpus'] = 19428  # hardcoded for OneStop (based on Adv)

        elif dataset_name == 'CopCo':
            data_parags = curr_df[
                ['part', 'unique_paragraph_id', 'paragraph']
            ].drop_duplicates()
            data_parags['n_words'] = data_parags['paragraph'].apply(
                lambda x: len((x).split())
            )
            stats['n_words_corpus'] = int(data_parags['n_words'].sum())
            df_to_save = data_parags[
                ['n_words', 'part', 'unique_paragraph_id', 'paragraph']
            ].sort_values(by=['part', 'unique_paragraph_id'])
            df_to_save.to_csv(STATS_FOLDER / 'CopCo_paragraphs.csv', index=False)
            logger.info(f'n_words calculated from {data_parags.shape[0]} paragraphs.')

        elif dataset_name == 'PoTeC':
            data_parags = (
                curr_df[['unique_paragraph_id', 'paragraph']]
                .drop_duplicates()
                .sort_values(by='unique_paragraph_id')
            )
            data_parags['n_words'] = data_parags['paragraph'].apply(
                lambda x: len((x).split())
            )
            stats['n_words_corpus'] = int(data_parags['n_words'].sum())
            logger.info(f'Total n_words: {stats["n_words_corpus"]}')
            df_to_save = data_parags[
                ['n_words', 'unique_paragraph_id', 'paragraph']
            ].sort_values(by='unique_paragraph_id')
            df_to_save.to_csv(
                STATS_FOLDER / f'{dataset_name}_paragraphs.csv', index=False
            )
            logger.info(f'n_words calculated from {data_parags.shape[0]} paragraphs.')

        elif dataset_name == 'SBSAT':
            data_parags = (
                curr_df[['unique_paragraph_id', 'paragraph']]
                .sort_values(by='unique_paragraph_id')
                .drop_duplicates()
                .sort_values(by='unique_paragraph_id')
            )
            data_parags['n_words'] = data_parags['paragraph'].apply(
                lambda x: len((x).split())
            )
            # TODO: fix bug in SBSAT data
            logger.warning(
                'using different caclulation because of BUG in SBSAT paragraphs'
            )
            grouped = (
                data_parags.groupby('unique_paragraph_id')[
                    ['unique_paragraph_id', 'n_words']
                ]
                .max()
                .reset_index(drop=True)
            )
            stats['n_words_corpus'] = int(grouped['n_words'].sum())
            logger.info(f'Total n_words: {stats["n_words_corpus"]}')
            df_to_save = data_parags[
                ['n_words', 'unique_paragraph_id', 'paragraph']
            ].sort_values(by='unique_paragraph_id')
            df_to_save.to_csv(
                STATS_FOLDER / f'{dataset_name}_paragraphs.csv', index=False
            )
            logger.info(f'n_words calculated from {data_parags.shape[0]} paragraphs.')

        elif dataset_name == 'MECOL2':
            data_parags = (
                curr_df[['paragraph', 'itemid', 'unique_paragraph_id']]
                .drop_duplicates()
                .sort_values(by='itemid')
                .reset_index(drop=True)
            )
            data_parags['n_words'] = data_parags['paragraph'].apply(
                lambda x: len((x).split())
            )
            # TODO: fix bug in MECO data
            logger.warning(
                'using different caclulation because of BUG in MECO paragraphs'
            )
            grouped = (
                data_parags.groupby('itemid')[['itemid', 'n_words']]
                .min()
                .reset_index(drop=True)
            )
            stats['n_words_corpus'] = int(grouped['n_words'].sum())
            logger.info(f'Total n_words: {stats["n_words_corpus"]}')
            df_to_save = data_parags[
                ['n_words', 'itemid', 'unique_paragraph_id', 'paragraph']
            ].sort_values(by='itemid')
            df_to_save.to_csv(
                STATS_FOLDER / f'{dataset_name}_paragraphs.csv', index=False
            )
            logger.info(f'n_words calculated from {data_parags.shape[0]} paragraphs.')

        elif dataset_name == 'IITBHGC':
            data_parags = (
                curr_df[['unique_paragraph_id', 'paragraph']]
                .drop_duplicates()
                .sort_values(by='unique_paragraph_id')
            )
            data_parags['n_words'] = data_parags['paragraph'].apply(
                lambda x: len((x).split())
            )
            stats['n_words_corpus'] = int(data_parags['n_words'].sum())
            logger.info(f'Total n_words: {stats["n_words_corpus"]}')
            df_to_save = data_parags[
                ['n_words', 'unique_paragraph_id', 'paragraph']
            ].sort_values(by='unique_paragraph_id')
            df_to_save.to_csv(
                STATS_FOLDER / f'{dataset_name}_paragraphs.csv', index=False
            )
            logger.info(f'n_words calculated from {data_parags.shape[0]} paragraphs.')

        else:
            logger.info(f'Unknown dataset: {dataset_name}')
            stats['n_words_corpus'] = '???'

        for task, label_col in task_and_labels.items():
            stats[f'{task}_col'] = label_col

            # if label_col is boolean, convert to int
            if pd.api.types.is_bool_dtype(curr_df[label_col]):
                curr_df[label_col] = curr_df[label_col].astype(int)

            # calc mean if not categorical
            if pd.api.types.is_numeric_dtype(curr_df[label_col]):
                stats[f'{task}_overall'] = format_stats(
                    trial_level[label_col], rounding=2
                )
                stats[f'{task}_per_participant'] = format_stats(
                    trial_level.groupby(participant_col)[label_col].mean(), rounding=2
                )
                stats[f'{task}_per_item'] = format_stats(
                    trial_level.groupby(item_col)[label_col].mean(), rounding=2
                )

            # label distribution if n unique values is small
            if trial_level[label_col].nunique() <= 15:
                label_distribution = (
                    trial_level[label_col].value_counts(normalize=True) * 100
                ).round(1)
                stats[f'{task}_distribution'] = label_distribution.to_dict()

    elif mode == DataType.FIXATIONS:
        stats = {
            'n_fix': curr_df.shape[0],
            'n_fix_per_trial': format_stats(curr_df.groupby('unique_trial_id').size()),
            'n_fix_per_participant': format_stats(
                curr_df.groupby(participant_col).size()
            ),
            'n_fix_per_item': format_stats(curr_df.groupby(item_col).size()),
        }

    elif mode == DataType.TRIAL_LEVEL:
        pass  # TODO: add?

    # save to stats all fields in additional_info_dataset except task_and_labels
    for key, val in additional_info.items():
        if key != 'task_and_labels':
            stats[key] = val

    # Load or create the profile DataFrame
    if output_path.exists():
        profile_df = pd.read_csv(output_path, index_col=0)
    else:
        profile_df = pd.DataFrame()

    # Initialize profile_df if empty
    if profile_df.empty:
        profile_df = pd.DataFrame(index=stats.keys(), columns=[data_args.dataset_name])
    else:
        for key in stats.keys():
            if key not in profile_df.index:
                profile_df.loc[key] = pd.Series()

    # Assign each stat
    for key, val in stats.items():
        profile_df.at[key, data_args.dataset_name] = val

    profile_df.sort_index(inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(output_path)

    logger.info(
        f'Updated profile for {data_args.dataset_name} ({mode}) → {output_path}'
    )


def calc_cols_stats_per_dataset(data_args, dataset_name):
    datasets_paths = {
        DataType.METADATA: ('.' / data_args.metadata_path),
        DataType.IA: ('.' / data_args.ia_path),
        DataType.FIXATIONS: ('.' / data_args.fixations_path),
    }

    for mode, path in datasets_paths.items():
        if not path.exists() or path.is_dir():
            logger.warning(
                f'{mode} - {path} does not exist or is a directory. Skipping...'
            )
            continue
        curr_df = load_raw_data(path)
        output_summary_path = (
            STATS_FOLDER / f'{data_args.dataset_name}_{mode}_summary.csv'
        )

        if curr_df is None or curr_df.empty:
            logger.warning(
                f'DataFrame for {data_args.dataset_name} {mode} is empty. Skipping...'
            )
            continue

        summary_df = summarize_dataframe(curr_df, dataset_name)

        if summary_df.empty:
            logger.warning(
                f'Summary DataFrame for {data_args.dataset_name} {mode} is empty. Skipping...'
            )
            continue

        summary_df.to_csv(output_summary_path, index=False)
        logger.info(f'Stats saved: {output_summary_path}')


def calc_task_stats_per_dataset(data_args, dataset_name):
    datasets_paths = {
        DataType.METADATA: ('.' / data_args.metadata_path),
        DataType.IA: ('.' / data_args.ia_path),
        DataType.FIXATIONS: ('.' / data_args.fixations_path),
    }

    for mode, path in datasets_paths.items():
        if not path.exists() or path.is_dir():
            logger.warning(
                f'{mode} - {path} does not exist or is a directory. Skipping...'
            )
            continue
        curr_df = load_raw_data(path)
        output_profile_path = STATS_FOLDER / 'tasks_stats.csv'

        if curr_df is None or curr_df.empty:
            logger.warning(
                f'DataFrame for {data_args.dataset_name} {mode} is empty. Skipping...'
            )
            continue

        compute_dataset_profile(
            dataset_name=dataset_name,
            mode=mode,
            curr_df=curr_df,
            output_path=output_profile_path,
            data_args=data_args,
        )


# TODO fold-level stats


def main(datasets: DataSets = DataSets) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    dataset = args.dataset

    if dataset:
        datasets = dataset.split(',')
    else:
        datasets = DataSets
    for dataset_name in datasets:
        data_args = get_data_args(dataset_name)
        if not data_args:
            logger.warning(f'No data args found for {dataset_name}. Skipping...')
            continue
        try:
            calc_cols_stats_per_dataset(data_args, dataset_name)
            calc_task_stats_per_dataset(data_args, dataset_name)
        except FileNotFoundError as e:
            logger.info(f'FileNotFoundError processing {dataset_name}: {e}')


if __name__ == '__main__':
    main()
