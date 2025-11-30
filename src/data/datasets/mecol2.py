"""Data module for creating the data."""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

from src.configs.constants import (
    SetNames,
)
from src.configs.main_config import Args
from src.data.datasets.base_dataset import ETDataset
from src.data.datasets.TextDataSet import TextDataSet

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # to avoid warnings


class MECOL2Dataset(ETDataset):
    """
    A PyTorch dataset for eye movement features.

    Args:
        cfg (Args): The configuration object.
        text_data (TextDataSet): The text data.
        ia_scaler (Union[MinMaxScaler, RobustScaler, StandardScaler]): Scaler for IA features.
        fixation_scaler (Union[MinMaxScaler, RobustScaler, StandardScaler, None]):
            Scaler for fixation features.
        trial_features_scaler (Union[MinMaxScaler, RobustScaler, StandardScaler, None]):
            The scaler for the trial features.
        regime_name (str, optional): The regime name. Defaults to "".
        set_name (str, optional): The set name. Defaults to "".
    """

    def __init__(
        self,
        cfg: Args,
        ia_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        fixation_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        trial_features_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        regime_name: SetNames,
        set_name: SetNames,
        text_data: TextDataSet | None = None,
    ):
        super().__init__(
            cfg=cfg,
            set_name=set_name,
            regime_name=regime_name,
            ia_scaler=ia_scaler,
            fixation_scaler=fixation_scaler,
            trial_features_scaler=trial_features_scaler,
            text_data=text_data,
        )

    def extract_trial_level_features(self) -> dict[str, torch.Tensor]:
        trial_level_features_list = []
        trial_level_features = self.trial_level_features.copy()
        trial_level_features = trial_level_features.drop(
            columns=self.ia_categorical_features,
            errors='ignore',
        )

        for grouped_data_key in tqdm(
            self.ordered_key_list, desc='Trial level features'
        ):
            try:
                trial_features = trial_level_features.loc[grouped_data_key]
            except:  # noqa: E722
                e1, e2, e3, e4 = grouped_data_key
                trial_features = trial_level_features.loc[(e4, e1, e2, e3)]

            trial_features = ETDataset.normalize_features(
                trial_features,
                normalize=self.normalize,
                scaler=self.trial_features_scaler,
            )
            trial_level_features_list.append(trial_features)

        return {
            'trial_level_features': torch.tensor(
                np.array(trial_level_features_list),
                dtype=torch.float32,
            )
        }
