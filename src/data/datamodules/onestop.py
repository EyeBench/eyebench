from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.configs.constants import SetNames
from src.data.datamodules.base_datamodule import ETDataModuleFast, register_datamodule
from src.data.datasets.onestop import OneStopDataset


@register_datamodule
class OneStopDataModule(ETDataModuleFast):
    """
    A PyTorch Lightning data module for the eye tracking data.

    Attributes:
        cfg (Args): The configuration object.
        text_dataset_path (Path): The path to the text dataset.
        train_dataset (OneStopDataSet): The training dataset.
        val_datasets (list[OneStopDataSet]): The validation datasets.
        test_datasets (list[OneStopDataSet]): The test datasets.
    """

    def create_etdataset(
        self,
        ia_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        fixation_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        trial_features_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        set_name: SetNames,
        regime_name: SetNames,
    ) -> OneStopDataset:
        """
        Create an OneStopDataSet instance for the given keys.

        Args:
            ia_scaler (MinMaxScaler | RobustScaler | StandardScaler): The IA scaler.
            fixation_scaler (MinMaxScaler | RobustScaler | StandardScaler | None): Fixation scaler.
            trial_features_scaler (MinMaxScaler | RobustScaler | StandardScaler | None):
                The trial features scaler.
            regime_name (SetNames): The name of the regime (e.g., unseen_subject_seen_item).
            set_name (SetNames): The name of the set (e.g., train, test, val).

        Returns:
            ETDataset: The created ETDataset instance.
        """
        text_data = None if self.cfg.model.use_eyes_only else self.load_text_dataset()

        dataset = OneStopDataset(
            cfg=self.cfg,
            ia_scaler=ia_scaler,
            fixation_scaler=fixation_scaler,
            trial_features_scaler=trial_features_scaler,
            regime_name=regime_name,
            set_name=set_name,
            text_data=text_data,
        )

        return dataset
