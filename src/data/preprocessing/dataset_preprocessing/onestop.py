from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tap import Tap
from text_metrics.utils import is_content_word
from tqdm import tqdm

from src.configs.constants import DataType, Fields
from src.configs.models.base_model import BaseModelArgs
from src.configs.models.dl.BEyeLSTM import BEyeLSTMArgs
from src.configs.models.dl.PLMASF import PLMASfArgs
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.data.utils import (
    add_missing_features,
    compute_trial_level_features,
)

logger.add(sink='logs/preprocessing.log', level='INFO')


IA_ID_COL = 'IA_ID'
FIXATION_ID_COL = 'CURRENT_FIX_INTEREST_AREA_INDEX'
NEXT_FIXATION_ID_COL = 'NEXT_FIX_INTEREST_AREA_INDEX'

tqdm.pandas()
logger.add('logs/preprocessing.log', level='INFO')


class OneStopProcessor(DatasetProcessor):
    """Processor for OneStop dataset"""

    def get_column_map(self, data_type: DataType) -> dict:
        """Get column mapping for OneStop dataset"""
        return {
            # Empty for now as it's handled in onestop processing
        }

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        return list(
            set(
                list(Fields)
                + BaseModelArgs().word_features
                + BaseModelArgs().eye_features
                + BaseModelArgs().fixation_features
                + BEyeLSTMArgs().fixation_features
                + BEyeLSTMArgs().eye_features
                + BEyeLSTMArgs().word_features
                + ['unique_trial_id']
                + self.data_args.groupby_columns
                + [
                    'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
                    'IA_FIRST_RUN_LANDING_POSITION',
                    'IA_LAST_RUN_LANDING_POSITION',
                    'NEXT_SAC_START_Y',
                    'NEXT_SAC_END_X',
                    'NEXT_SAC_END_Y',
                    'NEXT_SAC_START_X',
                    'ptb_pos',
                    'is_content_word',
                    'LengthCategory',
                    'is_reg_sum',
                    'is_progressive_sum',
                    'IA_REGRESSION_IN_COUNT_sum',
                    'normalized_outgoing_regression_count',
                    'normalized_outgoing_progressive_count',
                    'normalized_incoming_regression_count',
                    'LengthCategory_normalized_IA_DWELL_TIME',
                    'universal_pos_normalized_IA_DWELL_TIME',
                    'LengthCategory_normalized_IA_FIRST_FIXATION_DURATION',
                    'universal_pos_normalized_IA_FIRST_FIXATION_DURATION',
                    'entity_type',
                ]
            )
        )

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """OneStop-specific processing steps"""
        surprisal_models = ['gpt2']

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = data_dict[data_type]

            args = [
                '--mode',
                data_type,
                '--SURPRISAL_MODELS',
                *surprisal_models,
                '--onestopqa_path',
                str(self.data_args.onestopqa_path),
            ]
            cfg = ArgsParser().parse_args(args)
            if data_type == DataType.IA:
                df = self.query_onestop_data(df, query=self.data_args.ia_query)
            elif data_type == DataType.FIXATIONS:
                df = self.query_onestop_data(df, query=self.data_args.fixation_query)

            df = df.drop(columns=['ptb_pos']).rename(columns={'Reduced_POS': 'ptb_pos'})
            df = our_processing(df=df, args=cfg)

            # add unique_trial_id column
            df['unique_trial_id'] = (
                df['participant_id'].astype(str)
                + '_'
                + df['unique_paragraph_id'].astype(str)
                + '_'
                + df['repeated_reading_trial'].astype(str)
                + '_'
                + df['practice_trial'].astype(str)
            )

            # add is_correct column
            df['is_correct'] = (df.selected_answer == 'A').astype(int)

            df[Fields.LEVEL] = (
                df[Fields.LEVEL].replace({'Adv': 1, 'Ele': 0}).astype(int)
            )
            if data_type == DataType.IA:
                df['head_direction'] = df['distance_to_head'] > 0
                df['head_direction'] = df['head_direction'].astype(int)

            data_dict[data_type] = df

        data_dict['fixations'] = self.add_ia_report_features_to_fixation_data(
            data_dict['ia'], data_dict['fixations']
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )

        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        return data_dict

    def query_onestop_data(self, data: pd.DataFrame, query: str | None) -> pd.DataFrame:
        """Process the raw data by applying a query"""
        if query is not None:
            data = data.query(query)
            logger.info(f'Number of rows after query ({query}): {len(data)}')
        else:
            logger.info('***** No query! *****')
        return data

    def add_ia_report_features_to_fixation_data(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge per‑IA (interest‑area) features into the fixation‑level data.

        Result: one row per fixation, enriched with IA‑level attributes.
        """
        # --- 1. Unify IA‑ID column name ----------------------------------------
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # --- 2. Build the list of IA features we plan to add -------------------
        ia_features = (
            BEyeLSTMArgs().ia_features_to_add_to_fixation_data
            + BaseModelArgs().ia_features_to_add_to_fixation_data
            + PLMASfArgs().ia_features_to_add_to_fixation_data
            + ['entity_type']
        )

        required_cols = (
            self.data_args.groupby_columns
            + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
            + ia_features
        )
        ia_df = ia_df[list(set(required_cols))]

        # --- 3. Drop columns that also exist in fixation table -----------------
        merge_keys = set(
            self.data_args.groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
        )
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - merge_keys
        ia_df = ia_df.drop(columns=list(dup_cols))

        # --- 4. Clean nuisance column -----------------------------------------
        if 'normalized_part_ID' in fix_df.columns:
            if fix_df['normalized_part_ID'].isna().any():
                logger.warning('normalized_part_ID contains NaNs; dropping it.')
            fix_df = fix_df.drop(columns='normalized_part_ID')

        # --- 5. Merge ----------------------------------------------------------
        enriched_fix_df = fix_df.merge(
            ia_df,
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        num_of_words_in_trials_series = ia_df.groupby(
            self.data_args.groupby_columns,
        ).apply(len)
        num_of_words_in_trials_series.name = 'num_of_words_in_trial'
        merge_keys = set(self.data_args.groupby_columns)
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials_series,
            on=self.data_args.groupby_columns,
            how='left',
        )

        return enriched_fix_df


class Mode(Enum):
    """
    Enum for processing mode.
    Defines whether to process interest area (IA) or fixation data.
    """

    IA = 'ia'
    FIXATION = 'fixations'


class ArgsParser(Tap):
    """Args parser for preprocessing.py

        Note, for fixation data, the X_IA_DWELL_TIME, for X in
        [total, min, max, part_total, part_min, part_max]
        columns are computed based on the CURRENT_FIX_DURATION column.

        Note, documentation was generated automatically. Please check the source code for more info.
    Args:
        SURPRISAL_MODELS (list[str]): Models to extract surprisal from
        unique_item_columns (list[str]): columns that make up a unique item
         (Path | None): Path to question difficulty data from prolific
        mode (Mode): whether to use interest area or fixation data
    """

    SURPRISAL_MODELS: list[str] = [
        'gpt2',
    ]  # Models to shift surprisal for

    onestopqa_path: Path = Path('metadata/onestop_qa.json')
    mode: Mode = Mode.IA  # whether to use interest area or fixation data


def our_processing(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    LaCC lab-specific processing pipeline for OneStop dataset.

    Extends the public dataset with additional features including:
    - Integer and float feature conversions
    - Index adjustments
    - Fixation data cleaning
    - Unique paragraph ID addition
    - Word span metrics computation
    - Span-level metrics computation
    - Feature normalization
    - Question difficulty data integration
    - Previous word metrics (for IA mode)
    - Line position metrics (for IA mode)

    Args:
        df (pd.DataFrame): Input DataFrame from public preprocessing
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: Extended DataFrame with LaCC lab features
    """

    duration_field, ia_field = get_constants_by_mode(args.mode)

    df = convert_to_int_features(df, args)
    df = convert_to_float_features(df, args)
    df = adjust_indexing(df, args)
    df = drop_missing_fixation_data(df, args)
    df = add_unique_paragraph_id(df)
    df = compute_span_level_metrics(df, ia_field, args.mode, duration_field)
    df = compute_normalized_features(df, duration_field, ia_field)
    if args.mode == Mode.IA:
        df = compute_start_end_line(df)
        df = add_additional_metrics(df)

    return df


def convert_to_int_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to integer type.

    Handles missing values and dots by replacing them with 0 before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted integer columns
    """
    # In general, only features that have '.' or NaN or not automatically converted.

    to_int_features = [
        'article_batch',
        'article_id',
        'paragraph_id',
        'repeated_reading_trial',
        'practice_trial',
        # "question_preview",
    ]
    if args.mode == Mode.IA:
        to_int_features += [
            'IA_DWELL_TIME',
            'IA_FIRST_FIXATION_DURATION',
            'IA_REGRESSION_PATH_DURATION',
            'IA_FIRST_RUN_DWELL_TIME',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_RUN_COUNT',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_FIRST_RUN_FIXATION_COUNT',
            'IA_SKIP',
            'IA_REGRESSION_OUT_COUNT',
            'IA_SELECTIVE_REGRESSION_PATH_DURATION',
            'IA_SPILLOVER',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_LEFT',
            'IA_TOP',
            'TRIAL_DWELL_TIME',
            'TRIAL_FIXATION_COUNT',
            'TRIAL_IA_COUNT',
            'TRIAL_INDEX',
            'TRIAL_TOTAL_VISITED_IA_COUNT',
            'IA_FIRST_FIX_PROGRESSIVE',
        ]
    elif args.mode == Mode.FIXATION:
        to_int_features += [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_PUPIL',
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y',
            'CURRENT_FIX_INDEX',
            'NEXT_SAC_DURATION',
        ]
    df[to_int_features] = df[to_int_features].replace({'.': 0, np.nan: 0}).astype(int)
    logger.info(
        "%s fields converted to int, nan ('.') values replaced with 0.", to_int_features
    )
    return df


def convert_to_float_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to float type.

    Handles missing values and dots by replacing them with None before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted float columns
    """
    if args.mode == Mode.IA:
        to_float_features = [
            'IA_AVERAGE_FIX_PUPIL_SIZE',
            'IA_DWELL_TIME_%',
            'IA_FIXATION_%',
            'IA_FIRST_RUN_FIXATION_%',
            'IA_FIRST_SACCADE_AMPLITUDE',
            'IA_FIRST_SACCADE_ANGLE',
            'IA_LAST_RUN_FIXATION_%',
            'IA_LAST_SACCADE_AMPLITUDE',
            'IA_LAST_SACCADE_ANGLE',
            'IA_FIRST_RUN_LANDING_POSITION',
            'IA_LAST_RUN_LANDING_POSITION',
        ]
    elif args.mode == Mode.FIXATION:
        to_float_features = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            'NEXT_FIX_ANGLE',
            'PREVIOUS_FIX_ANGLE',
            'NEXT_FIX_DISTANCE',
            'PREVIOUS_FIX_DISTANCE',
            'NEXT_SAC_AMPLITUDE',
            'NEXT_SAC_ANGLE',
            'NEXT_SAC_AVG_VELOCITY',
            'NEXT_SAC_PEAK_VELOCITY',
            'NEXT_SAC_END_X',
            'NEXT_SAC_START_X',
            'NEXT_SAC_END_Y',
            'NEXT_SAC_START_Y',
        ]
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
    df[to_float_features] = (
        df[to_float_features].replace(to_replace={'.': None}).astype(float)
    )
    logger.info(
        "%s fields converted to float, nan ('.') values replaced with None.",
        to_float_features,
    )
    return df


def adjust_indexing(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Adjust indexing to be 0-indexed.

    Subtracts 1 from specified columns based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with adjusted indexing
    """
    if args.mode == Mode.IA:
        subtract_one_fields = [IA_ID_COL]
    elif args.mode == Mode.FIXATION:
        subtract_one_fields = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
        ]
    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    df[subtract_one_fields] -= 1
    logger.info('%s values adjusted to be 0-indexed.', subtract_one_fields)
    return df


def drop_missing_fixation_data(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Drop rows with missing fixation data.

    Drops rows with missing values in specified columns for FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with dropped rows
    """
    if args.mode == Mode.FIXATION:
        dropna_fields = [FIXATION_ID_COL, NEXT_FIXATION_ID_COL]
        df = df.dropna(subset=dropna_fields)
        logger.info(
            'After dropping rows with missing data in %s: %d records left in total.',
            dropna_fields,
            len(df),
        )
    return df


def add_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional metrics to the DataFrame.

    Adds columns for regression rate, total skip, and part length.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added metrics
    """

    logger.info('Adding additional metrics...')
    df['regression_rate'] = df['IA_REGRESSION_OUT_FULL_COUNT'] / df['IA_RUN_COUNT']
    df['total_skip'] = df['IA_DWELL_TIME'] == 0
    df['is_content_word'] = df['universal_pos'].apply(is_content_word)

    return df


def get_constants_by_mode(mode: Mode) -> tuple[str, str]:
    """
    Get constants based on processing mode.

    Returns duration and IA field names based on whether in IA or FIXATION mode.

    Args:
        mode (Mode): Processing mode (IA or FIXATION)

    Returns:
        tuple[str, str]: Duration and IA field names
    """
    duration_field = 'IA_DWELL_TIME' if mode == Mode.IA else 'CURRENT_FIX_DURATION'
    ia_field = IA_ID_COL if mode == Mode.IA else FIXATION_ID_COL

    return duration_field, ia_field


def add_unique_paragraph_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add unique paragraph ID to the DataFrame.

    Creates a new column 'unique_paragraph_id' by combining article_batch,
    article_id, difficulty_level, and paragraph_id.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added unique paragraph ID
    """
    logger.info('Adding unique paragraph id...')
    df['unique_paragraph_id'] = (
        df[['article_batch', 'article_id', 'difficulty_level', 'paragraph_id']]
        .astype(str)
        .apply('_'.join, axis=1)
    )
    return df


def get_raw_text(args: object) -> dict:
    """
    Load raw text data from OneStopQA JSON file.

    Args:
        args: Configuration containing onestopqa_path

    Returns:
        dict: Raw text data from OneStopQA JSON
    """
    with open(
        file=args.onestopqa_path,
        mode='r',
        encoding='utf-8',
    ) as f:
        raw_text = json.load(f)
    return raw_text['data']


def get_article_data(article_id: str, raw_text) -> dict:
    """
    Retrieve article data from raw text by article ID.

    Args:
        article_id (str): Article identifier to look up
        raw_text (dict): Raw text data containing articles

    Returns:
        dict: Article data if found

    Raises:
        ValueError: If article ID not found
    """
    for article in raw_text:
        if article['article_id'] == article_id:
            return article
    raise ValueError(f'Article id {article_id} not found')


def compute_start_end_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute for each word  whether it the first/last word in the line (not sentence!).

    This function adds two new columns to the input DataFrame: 'start_of_line' and 'end_of_line'.
    A word is considered to be at the start of a line if its
        'IA_LEFT' value is smaller than the previous word's.
    A word is considered to be at the end of a line if its
        'IA_LEFT' value is larger than the next word's.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must contain the columns 'participant_id',
        'unique_paragraph_id', and 'IA_LEFT'.

    Returns:
    pd.DataFrame: The input DataFrame with two new columns: 'start_of_line' and 'end_of_line'.
    """

    logger.info('Adding start_of_line and end_of_line columns...')
    grouped_df = df.groupby(
        ['participant_id', 'unique_paragraph_id', 'repeated_reading_trial']
    )
    df['start_of_line'] = (
        grouped_df['IA_LEFT'].shift(periods=1, fill_value=1000000) > df['IA_LEFT']
    )
    df['end_of_line'] = (
        grouped_df['IA_LEFT'].shift(periods=-1, fill_value=-1) < df['IA_LEFT']
    )
    return df


def compute_span_level_metrics(
    df: pd.DataFrame, ia_field: str, mode: Mode, duration_col: str
) -> pd.DataFrame:
    """
    Calculate aggregated metrics for different text spans.

    Computes:
    - Total dwell time per trial/span
    - Min/max word indices per trial/span
    - For fixations: count per span
    - Normalizes indices to start at 0

    Args:
        df (pd.DataFrame): Input DataFrame
        ia_field (str): Column name for word/fixation index
        mode (Mode): IA or FIXATION processing mode
        duration_col (str): Column name for duration values

    Returns:
        pd.DataFrame: DataFrame with added span-level metrics
    """
    logger.info('Computing span-level metrics...')

    group_by_fields = [
        'participant_id',
        'unique_paragraph_id',
        'repeated_reading_trial',
    ]

    # Fix trials where ID does not start at 0
    if mode == Mode.IA:
        temp_max_per_trial = df.groupby(group_by_fields).agg(
            min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='min'),
            max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='max'),
        )
        non_zero_min_ia_id_trials = temp_max_per_trial[
            temp_max_per_trial['min_IA_ID'] != 0
        ]
        logger.info(
            'Number of trials where min_IA_ID is not zero: %d out of %d trials.',
            len(non_zero_min_ia_id_trials),
            len(temp_max_per_trial),
        )
        df = df.merge(
            temp_max_per_trial,
            on=group_by_fields,
            validate='m:1',
            suffixes=(None, '_y'),
        )
        logger.info('Shifting IA_ID to start at 0...')
        df[ia_field] -= df['min_IA_ID']
        df.drop(columns=['min_IA_ID', 'max_IA_ID'], inplace=True)

    max_per_trial = df.groupby(group_by_fields).agg(
        total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc='sum'),
        min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='min'),
        max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='max'),
    )
    df = df.merge(
        max_per_trial, on=group_by_fields, validate='m:1', suffixes=(None, '_y')
    )
    return df


def compute_normalized_features(
    df: pd.DataFrame, duration_col: str, ia_field: str
) -> pd.DataFrame:
    """
    Calculate normalized versions of key metrics.

    Adds columns for:
    - Normalized dwell times (total and by part)
    - Normalized word positions (total and by part)
    - Reverse indices from end

    Args:
        df (pd.DataFrame): Input DataFrame
        duration_col (str): Column name for duration values
        ia_field (str): Column name for word/fixation index

    Returns:
        pd.DataFrame: DataFrame with normalized metrics
    """
    logger.info('Computing normalized dwell time, and normalized word indices...')
    df = df.assign(
        normalized_ID=(df[ia_field] - df.min_IA_ID) / (df.max_IA_ID - df.min_IA_ID),
    ).copy()
    return df
