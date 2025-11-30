"""Data module for creating the data."""

from __future__ import annotations

import os
import pickle
import warnings
from abc import abstractmethod
from dataclasses import asdict
from typing import Type

import lightning.pytorch as pl
from loguru import logger
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader

from src.configs.constants import FEATURES_CACHE_FOLDER, REGIMES, Scaler, SetNames
from src.configs.main_config import Args
from src.data.datasets.base_dataset import ETDataset
from src.data.datasets.sbsat import TextDataSet

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # to avoid warnings


class DataModuleFactory:
    datamodules = {}

    @classmethod
    def add(cls, datamodule: Type[ETDataModuleFast]) -> None:
        cls.datamodules[datamodule.__name__] = datamodule
        # print(f'Registered datamodule: {datamodule.__name__}')

    @classmethod
    def get(cls, datamodule_name: str) -> Type[ETDataModuleFast]:
        datamodule = cls.datamodules[datamodule_name]
        return datamodule


def register_datamodule(datamodule: Type[ETDataModuleFast]) -> Type[ETDataModuleFast]:
    DataModuleFactory.add(datamodule)
    return datamodule


class ETDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for the eye tracking data.

    Attributes:
        cfg (Args): The configuration object.
        text_dataset_path (Path): The path to the text dataset.
        train_dataset (ETDataset): The training dataset.
        val_datasets (list[ETDataset]): The validation datasets.
        test_datasets (list[ETDataset]): The test datasets.
    """

    def __init__(self, cfg: Args):
        """
        Initialize the ETDataModule instance.

        Args:
            cfg (Args): The configuration object.
        """
        super().__init__()
        self.cfg = cfg

        self.train_dataset: ETDataset
        self.val_datasets: list[ETDataset]
        self.test_datasets: list[ETDataset]

        self.text_dataset_path = (
            FEATURES_CACHE_FOLDER
            / f'{cfg.data.dataset_name}_{cfg.data.task}_{cfg.model.model_name}'
            / 'TextDataSet.pkl'
        )

        self.save_hyperparameters(asdict(self.cfg))

    def setup(self, stage: str | None = None) -> None:
        """
        Set up the data module for training, validation, or testing.

        Args:
            stage (str | None): The stage of the setup. Can be "fit", "test", or "predict".
        """

        ia_scaler = self.cfg.model.normalization_type.value()
        fixation_scaler = self.cfg.model.normalization_type.value()
        trial_features_scaler = self.cfg.model.normalization_type.value()

        self.train_dataset = self.create_etdataset(
            ia_scaler=ia_scaler,
            fixation_scaler=fixation_scaler,
            trial_features_scaler=trial_features_scaler,
            set_name=SetNames.TRAIN,
            regime_name=SetNames.TRAIN,
        )

        if stage in {'fit', 'predict'}:
            self.val_datasets = [
                self.create_etdataset(
                    ia_scaler=self.train_dataset.ia_scaler,
                    fixation_scaler=self.train_dataset.fixation_scaler,
                    trial_features_scaler=self.train_dataset.trial_features_scaler,
                    regime_name=regime_name,
                    set_name=SetNames.VAL,
                )
                for regime_name in REGIMES
            ]

        if stage in {'test', 'predict'}:
            self.test_datasets = [
                self.create_etdataset(
                    ia_scaler=self.train_dataset.ia_scaler,
                    fixation_scaler=self.train_dataset.fixation_scaler,
                    trial_features_scaler=self.train_dataset.trial_features_scaler,
                    regime_name=regime_name,
                    set_name=SetNames.TEST,
                )
                for regime_name in REGIMES
            ]

    @abstractmethod
    def create_etdataset(
        self,
        ia_scaler: Scaler | None,
        fixation_scaler: Scaler | None,
        trial_features_scaler: Scaler | None,
        set_name: SetNames,
        regime_name: SetNames,
    ) -> ETDataset:
        """
        Abstract method to create an ETDataset instance.

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
        raise NotImplementedError('Subclasses must implement this method.')

    def create_dataloader(
        self,
        dataset,
        shuffle,
        sample_m_per_class: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset (ETDataset): The dataset to create the DataLoader for.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: The created DataLoader.
        """
        if sample_m_per_class:
            sampler = samplers.MPerClassSampler(
                labels=self.train_dataset.labels,
                m=1,
                length_before_new_iter=self.cfg.trainer.samples_per_epoch,
            )
            shuffle = None
            logger.info(
                f'Using MPerClassSampler with m=1 and {self.cfg.trainer.samples_per_epoch} samples per epoch. Shuffle is set to None.'
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.cfg.model.batch_size,
            num_workers=self.cfg.trainer.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=drop_last,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        return self.create_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=False,
            sample_m_per_class=self.cfg.trainer.sample_m_per_class,
        )

    def val_dataloader(self) -> list[DataLoader]:
        """
        Create the DataLoader for the validation datasets.

        Returns:
            list[DataLoader]: A list of DataLoaders for the validation datasets.
        """
        return [
            self.create_dataloader(dataset, shuffle=False, drop_last=False)
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> list[DataLoader]:
        """
        Create the DataLoader for the test datasets.

        Returns:
            list[DataLoader]: A list of DataLoaders for the test datasets.
        """
        return [
            self.create_dataloader(dataset, shuffle=False, drop_last=False)
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        """
        Create the DataLoader for the prediction datasets.

        Returns:
            list[DataLoader]: A list of DataLoaders for the prediction datasets.
        """
        return self.val_dataloader() + self.test_dataloader()

    def prepare_data(self) -> None:
        """
        Prepare the data for the module.

        """

        self.text_dataset_create_if_needed()

    def text_dataset_create_if_needed(self) -> None:
        """
        If the text dataset does not exist or overwrite_data is True, create and save the text dataset.
        """

        if self.cfg.model.use_eyes_only:
            logger.info('Using eyes only, no text dataset will be created.')
            return

        if self.cfg.trainer.overwrite_data or not self.text_dataset_path.exists():
            self.text_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Creating and saving textDataSet to {self.text_dataset_path}')
            # create and save to pkl
            text_data = TextDataSet(cfg=self.cfg)
            with open(self.text_dataset_path, 'wb') as f:
                pickle.dump(text_data, f)
        else:
            logger.info(
                f'TextDataSet already exists at: {self.text_dataset_path} and overwrite is False'
            )

    def load_text_dataset(self) -> TextDataSet:
        """
        Load the text dataset from a pickle file.

        Returns:
            TextDataSet: The loaded text dataset.
        """
        logger.info(f'Loading textDataSet from {self.text_dataset_path}')
        with open(self.text_dataset_path, 'rb') as f:
            text_data = pickle.load(f)
        return text_data


class ETDataModuleFast(ETDataModule):
    """
    A subclass of ETDataModule that includes checks to prevent redundant data preparation and setup.
    Based on the solution provided in https://github.com/Lightning-AI/pytorch-lightning/issues/16005

    Attributes:
        prepare_data_done (bool): A flag indicating whether the prepare_data method has been called.
        setup_stages_done (set): A set storing the stages for which setup method has been called.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """
        Initialize the ETDataModuleFast instance.

        Args:
            *args: Variable length argument list to be passed to the ETDataModule constructor.
            **kwargs: Arbitrary keyword arguments to be passed to the ETDataModule constructor.
        """
        super().__init__(*args, **kwargs)
        self.prepare_data_done = False
        self.setup_stages_done = set()

    def prepare_data(self) -> None:
        """
        Prepare data for the module. If this method has been called before, it does nothing.
        """
        if not self.prepare_data_done:
            super().prepare_data()
            self.prepare_data_done = True

    def setup(self, stage: str) -> None:
        """
        Set up the module for a specific stage.
            If this method has been called before for the same stage, it does nothing.

        Args:
            stage (str): The stage for which to set up the module.
        """
        if stage not in self.setup_stages_done:
            super().setup(stage)
            self.setup_stages_done.add(stage)
