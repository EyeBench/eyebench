from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
import pymovements as pm
from loguru import logger

from src.configs.constants import STATS_FOLDER, DataSets
from src.configs.data import get_data_args
from src.data.preprocessing.stats import summarize_dataframe

logger.add('logs/preprocessing.log', level='INFO')


def combine_files(
    dataset: list[pm.reading_measures.ReadingMeasures | pl.DataFrame],
    fileinfo: list[pl.DataFrame],
    output_csv: Path,
    output_summary_csv: Path,
    dataset_name: str,
) -> None:
    # Combine all precomputed events into a single DataFrame
    if len(dataset) > 1:
        logger.info('Merging...')
        combined_df = pd.concat(
            [
                # Combine each frame with its corresponding fileinfo
                pd.concat(
                    [
                        frame.frame.to_pandas().rename(
                            {
                                f: f.replace('"', '')
                                for f in frame.frame.to_pandas().columns
                            },
                            axis=1,
                        ),
                        pd.DataFrame(
                            {
                                'source_file': [
                                    subject_id for _ in range(len(frame.frame))
                                ]
                            }
                        ),
                    ],
                    axis=1,
                )
                for frame, subject_id in zip(dataset, fileinfo['subject_id'])
            ],
            ignore_index=True,
        )
    else:
        logger.info('Only one datafile...')
        combined_df = dataset[0].frame.to_pandas()

    logger.info(f'Total combined rows: {len(combined_df)}')

    # Save combined CSV
    combined_df.to_csv(output_csv, index=False)
    logger.info(f'Combined CSV saved: {output_csv}')

    # Save column summary
    summary_df = summarize_dataframe(combined_df, dataset_name)
    summary_df.to_csv(output_summary_csv, index=False)
    logger.info(f'Column summary saved: {output_summary_csv}')


def combine_stimulus_files(
    data_path: Path,
    matching_pattern: str,
    dataset_name: str,
) -> None:
    """
    Combine stimulus files from a given data path matching a specific pattern.

    Args:
        data_path (str): Path to the directory containing stimulus files.
        matching_pattern (str): Pattern to match files for combining.
    """
    stimulus_files = list(data_path.rglob(matching_pattern))

    if not stimulus_files:
        logger.warning(f'No files found matching {matching_pattern} in {data_path}.')
        return

    combined_df = pd.DataFrame()
    for file_ in stimulus_files:
        if 'reading' in str(file_):
            read_df = pd.read_csv(file_, sep='\t')
            read_df['filename'] = f'{file_.name.split("_")[0]}.png'
            read_df['sequence_num'] = -1
            combined_df = pd.concat([combined_df, read_df])
        else:
            quest_df = pd.read_csv(file_)
            quest_df['sequence_num'] = file_.name.split('_')[0].split('-')[-1]
            combined_df = pd.concat([combined_df, quest_df])

    if 'SBSAT' in str(data_path):
        logger.info('Filling in missing values for SBSAT dataset...')
        combined_df['stimulus_type'] = combined_df['filename'].apply(
            lambda x: x.split('-')[1]
        )
        combined_df['is_question'].fillna(False, inplace=True)

        # Create 'question' column by concatenating all 'word' where 'is_question' is True for each filename
        question_map = (
            combined_df[combined_df['is_question']]
            .groupby(['stimulus_type', 'sequence_num'])['word']
            .apply(' '.join)
        )
        combined_df['question'] = combined_df.apply(
            lambda row: question_map.get(
                (row['stimulus_type'], row['sequence_num']), None
            ),  # type: ignore
            axis=1,
        )

    combined_df.to_csv(Path(data_path) / 'combined_stimulus.csv', index=False)
    logger.info(f'Combined stimulus CSV saved: {data_path / "combined_stimulus.csv"}')


def combine_dataset(dataset_name: str) -> None:
    logger.info(f'Processing {dataset_name}...')
    if dataset_name == 'MECOL2':
        lookup = {}
        for part in ('W1', 'W2'):
            lookup[f'data_args_{part}'] = get_data_args(f'{dataset_name}{part}')
            base = lookup[f'data_args_{part}'].base_path
            logger.info(f'Processing {dataset_name}{part}...')
            dataset_def = pm.DatasetLibrary.get(f'{dataset_name}{part}')
            dataset_def.has_files['gaze'] = False
            logger.info(f'Loading {dataset_name}{part} dataset...')
            dataset = pm.Dataset(dataset_def, f'data/{dataset_name}{part}').load()
            if part == 'W1':
                dataset.precomputed_events[0].frame = (
                    dataset.precomputed_events[0]
                    .frame.to_pandas()
                    .drop('trial', axis=1)
                )
                dataset.precomputed_reading_measures[
                    0
                ].frame = dataset.precomputed_reading_measures[0].frame.to_pandas()
            if part == 'W2':
                dataset.precomputed_events[0].frame = dataset.precomputed_events[
                    0
                ].frame.to_pandas()
                dataset.precomputed_reading_measures[0].frame = (
                    dataset.precomputed_reading_measures[0]
                    .frame.to_pandas()
                    .drop('supplementary_id', axis=1)
                )
            lookup[f'dataset_{part}'] = dataset
        fix = pd.concat(
            [
                lookup['dataset_W1'].precomputed_events[0].frame,
                lookup['dataset_W2'].precomputed_events[0].frame,
            ],
            ignore_index=True,
        )
        ia = pd.concat(
            [
                lookup['dataset_W1'].precomputed_reading_measures[0].frame,
                lookup['dataset_W2'].precomputed_reading_measures[0].frame,
            ],
            ignore_index=True,
        )
        base = Path('data/MECOL2')
        base.mkdir(parents=True, exist_ok=True)
        Path(base / 'precomputed_events').mkdir(parents=True, exist_ok=True)
        Path(base / 'precomputed_reading_measures').mkdir(parents=True, exist_ok=True)
        logger.info(f'Total combined rows precomputed events: {len(fix)}')
        output_csv = base / 'precomputed_events' / 'combined_fixations.csv'
        output_summary_csv = STATS_FOLDER / f'{dataset_name}_raw_fixations_summary.csv'
        fix.to_csv(output_csv, index=False)
        logger.info(f'Combined CSV saved: {output_csv}')
        summary_df = summarize_dataframe(fix, dataset_name)
        summary_df.to_csv(output_summary_csv, index=False)
        logger.info(f'Column summary saved: {output_summary_csv}')
        logger.info(f'Total combined rows precomputed reading measures: {len(ia)}')
        output_csv = base / 'precomputed_reading_measures' / 'combined_ia.csv'
        output_summary_csv = STATS_FOLDER / f'{dataset_name}_raw_ia_summary.csv'
        ia.to_csv(output_csv, index=False)
        logger.info(f'Combined CSV saved: {output_csv}')
        summary_df = summarize_dataframe(ia, dataset_name)
        summary_df.to_csv(output_summary_csv, index=False)
        logger.info(f'Column summary saved: {output_summary_csv}')

    else:
        data_args = get_data_args(dataset_name)
        base = data_args.base_path

        dataset_def = pm.DatasetLibrary.get(dataset_name)
        dataset_def.has_files['gaze'] = False

        logger.info(f'Loading {dataset_name} dataset...')
        dataset = pm.Dataset(dataset_def, f'data/{dataset_name}').load()

        if dataset.definition.has_files['precomputed_events']:
            logger.info('Processing precomputed events...')
            combine_files(
                dataset=dataset.precomputed_events,
                fileinfo=dataset.fileinfo['precomputed_events'],
                output_csv=base / 'precomputed_events' / 'combined_fixations.csv',
                output_summary_csv=STATS_FOLDER
                / f'{dataset_name}_raw_fixations_summary.csv',
                dataset_name=dataset_name,
            )
        else:
            logger.info(f'{dataset_name} has no precomputed events...')

        if dataset.definition.has_files['precomputed_reading_measures']:
            logger.info('Processing precomputed reading measures...')
            combine_files(
                dataset=dataset.precomputed_reading_measures,
                fileinfo=dataset.fileinfo['precomputed_reading_measures'],
                output_csv=base / 'precomputed_reading_measures' / 'combined_ia.csv',
                output_summary_csv=STATS_FOLDER / f'{dataset_name}_raw_ia_summary.csv',
                dataset_name=dataset_name,
            )
        else:
            logger.info(f'{dataset_name} has no precomputed reading measures...')

        if dataset_name == 'SBSAT':
            if data_args.raw_ia_dir:
                logger.info('Combining stimulus files...')
                combine_stimulus_files(
                    data_path=data_args.raw_ia_dir,
                    matching_pattern='*words.csv',
                    dataset_name=dataset_name,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='')
    args = parser.parse_args()

    if args.dataset:
        datasets = args.dataset.split(',')
    else:
        datasets = DataSets

    for dataset_name in datasets:
        combine_dataset(dataset_name)


if __name__ == '__main__':
    raise SystemExit(main())
