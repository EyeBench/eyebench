from __future__ import annotations

import pandas as pd

from src.configs.constants import DataType
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor


class TemplateProcessor(DatasetProcessor):
    """Processor for DATASET_NAME dataset"""

    def get_column_map(self, data_type: DataType) -> dict:
        """Get column mapping for DATASET_NAME dataset"""
        # TODO: add docs
        if data_type == DataType.IA:
            return {}
        elif data_type == DataType.FIXATIONS:
            return {}

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        # TODO: add docs
        return []

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Dataset-specific processing steps"""
        # TODO: add docs
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue
            # load data
            df = data_dict[data_type]

            # add ids
            # add unique_trial_id column
            df['unique_trial_id'] = (
                df['participant_id'].astype(str)
                + '_'
                + df['unique_paragraph_id'].astype(str)
                + '_'
                + df['practice_trial'].astype(str)
            )
            # filter rows?
            # add labels of tasks?

            data_dict[data_type] = df

        # add_ia_report_features_to_fixation_data ?
        # add_missing_features ?
        # compute_trial_level_features ?

        return data_dict

    def add_ia_report_features_to_fixation_data(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        # TODO: add docs
        #     # --- 1. Unify IA‑ID column name ----------------------------------------
        #     # --- 2. Build the list of IA features we plan to add -------------------
        #     # --- 3. Drop columns that also exist in fixation table -----------------
        #     # --- 4. Clean nuisance column ------------------------------------------
        #     # --- 5. Merge ----------------------------------------------------------
        #     return
        """
