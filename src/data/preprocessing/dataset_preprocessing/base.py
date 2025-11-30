from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

from src.configs.constants import DataType
from src.configs.data import DataArgs
from src.data.utils import (
    load_raw_data,
)

logger.add('logs/preprocessing.log', level='INFO')


class DatasetProcessor:
    """Base class for dataset processors"""

    def __init__(self, data_args: DataArgs):
        self.data_args = data_args

    def process(self) -> dict[str, pd.DataFrame]:
        """Process the dataset"""
        # Load raw data
        raw_data = {}
        if self.data_args.raw_ia_path:
            raw_data[DataType.IA] = self.load_raw_data(self.data_args.raw_ia_path)
        else:
            raw_data[DataType.IA] = None

        if self.data_args.raw_fixations_path:
            raw_data[DataType.FIXATIONS] = self.load_raw_data(
                self.data_args.raw_fixations_path
            )
        else:
            raw_data[DataType.FIXATIONS] = None

        # Standardize column names
        processed_data = {}
        for data_type, df in raw_data.items():
            if df is not None:
                processed_data[data_type] = self.standardize_column_names(
                    df, data_type=data_type
                )

        # Dataset-specific processing
        processed_data = self.dataset_specific_processing(processed_data)

        # Filter data
        for data_type in processed_data:
            # Trial level are computed, so we want to keep them as is
            if data_type != DataType.TRIAL_LEVEL:
                processed_data[data_type] = self.filter_data(processed_data[data_type])

        return processed_data

    def load_raw_data(self, data_path: Path) -> pd.DataFrame:
        return load_raw_data(data_path)

    def standardize_column_names(
        self, df: pd.DataFrame, data_type: DataType
    ) -> pd.DataFrame:
        """Standardize column names for the dataset"""
        # Map of original column names to standardized names
        column_map = self.get_column_map(data_type)

        if column_map:
            # Create a dictionary of only the columns that exist in the dataframe
            valid_columns = {k: v for k, v in column_map.items() if k in df.columns}
            logger.info(
                f'Standardizing column names for {self.data_args.dataset_name} {data_type}: {valid_columns}'
            )
            return df.rename(columns=valid_columns)

        logger.info(
            f'{self.data_args.dataset_name} not found in column maps. No changes made.'
        )
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on dataset-specific criteria"""
        columns_to_keep = self.get_columns_to_keep()

        if columns_to_keep:
            missing_columns = [col for col in columns_to_keep if col not in df.columns]
            if missing_columns:
                logger.warning(
                    f'Missing columns not found in the dataframe: {missing_columns}'
                )
                logger.warning(f'All possible columns: {list(df.columns)}')
            existing_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[existing_columns]
            logger.info(
                f'Filtering columns for {self.data_args.dataset_name}: {existing_columns}'
            )

        return df

    def save_processed_data(self, processed_data: dict[str, pd.DataFrame]) -> None:
        """
        Save processed data to processed data folder.

        Args:
            processed_data: Dictionary containing processed dataframes
        """
        self.data_args.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Save each dataframe to the processed folder
        for data_type, df in processed_data.items():
            if df is not None:
                output_path = (
                    self.data_args.processed_data_path / f'{data_type}.feather'
                )
                df.to_feather(output_path)
                logger.info(f'Saved {data_type} to {output_path}')

    @abstractmethod
    def get_column_map(self, data_type: DataType) -> dict:
        """Get column mapping for the dataset"""
        return {}

    @abstractmethod
    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        return []

    @abstractmethod
    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Dataset-specific processing steps"""
        # Can use the following for surprisal and other metric calculations
        # from text_metrics.merge_metrics_with_eye_movements import (
        #     add_metrics_to_word_level_eye_tracking_report,
        # )
        # from text_metrics.surprisal_extractors import extractor_switch
        return data_dict
