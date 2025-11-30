"""
This script is responsible for splitting datasets into training, validation, and test sets.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.configs.constants import REGIMES, DataSets, DataType, Fields, SetNames
from src.configs.data import DataArgs, get_data_args
from src.data.utils import load_raw_data

logger.add('logs/preprocessing.log', level='INFO')


class FoldSplitter:
    """
    A class used to split data into folds.

    Attributes:
        item_columns (list[str]): The columns that contain the item identifiers.
        subject_column (str): The column that contains the subject identifiers.
        groupby_columns (list[str]): The columns used to group the trials.
        num_folds (int): The number of folds to split the data into.
        stratify (str | None): The column that contains the target values.
        folds_path (Path): The path where the fold CSVs are stored.
    """

    def __init__(
        self,
        item_columns: list[str],
        subject_column: str,
        groupby_columns: list[str],
        num_folds: int,
        stratify: str | None,
        folds_path: Path = Path('data') / 'folds',
        higher_level_split_column: str | None = None,
    ) -> None:
        """
        Initialize the FoldSplitter.

        Parameters:
            item_columns (list[str]): The columns that contain the item identifiers.
            subject_column (str): The column that contains the subject identifiers.
            groupby_columns (list[str]): The columns used to group the trials.
            higher_level_split_column (str): The column used for higher level splitting.
            num_folds (int): The number of folds to split the data into.
            stratify (str): The column that contains the target values.
            folds_path (Path, optional): The path where the fold CSVs are stored.
        """
        self.item_columns = item_columns
        self.subject_column = subject_column
        self.groupby_columns = groupby_columns
        self.higher_level_split_column = higher_level_split_column
        self.num_folds = num_folds
        self.stratify = stratify
        self.folds_path = folds_path
        self.item_folds = {}
        self.subject_folds = {}

    def get_split_indices(
        self, group_keys: pd.DataFrame, split_indices: pd.Series, is_item: bool
    ) -> pd.Index:
        """
        Get the indices from group keys based on item or subject split.

        Parameters:
            group_keys (pd.DataFrame): DataFrame containing grouping keys.
            split_indices (pd.Series): Series of split indices.
            is_item (bool): Whether the split is for item identifiers.

        Returns:
            pd.Index: The indices from group_keys that match the split.
        """
        if is_item:
            column = group_keys[self.item_columns].astype(str).apply('_'.join, axis=1)
        else:
            column = group_keys[self.subject_column]

        return group_keys.loc[column.isin(split_indices)].index

    def load_folds(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Load folds from the CSV files for subjects and items.

        Returns:
            tuple[list[pd.DataFrame], list[pd.DataFrame]]: A tuple containing the subject folds
                and item folds as DataFrames.
        """
        try:
            folds_path = Path(HydraConfig.get().runtime.output_dir) / 'folds'
        except Exception:  # no hydra
            logger.info(
                f'HydraConfig not found. Using default path. Loading folds from {self.folds_path}'
            )
            folds_path = self.folds_path

        subject_folds_path = folds_path / 'subjects'
        item_folds_path = folds_path / 'items'
        # load all folds

        for i in range(self.num_folds):
            subject_fold_path = subject_folds_path / f'fold_{i}.csv'
            item_fold_path = item_folds_path / f'fold_{i}.csv'
            self.subject_folds[i] = pd.read_csv(subject_fold_path, header=None)
            self.item_folds[i] = pd.read_csv(item_fold_path, header=None)
        return self.subject_folds, self.item_folds

    def get_fold_indices(
        self,
        i: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Get fold indices for test, validation, and training sets based on the given fold index.

        Parameters:
            i (int): The fold index (should be between 0 and num_folds - 1).

        Returns:
            tuple[list[int], list[int], list[int]]: A tuple containing the test indices,
                validation indices, and training indices.
        """
        if i < 0 or i > self.num_folds - 1:
            raise ValueError('Fold index must be within the range [0, num_folds - 1].')

        validation_indices = [i]
        # modulo num_folds for the wraparound
        test_indices = [(i + 1) % self.num_folds]

        # The rest are training indices
        train_indices = [
            x
            for x in range(self.num_folds)
            if x not in test_indices and x not in validation_indices
        ]
        logger.info(
            f'Test folds: {test_indices}, Validation fold: {validation_indices}, Train folds: {train_indices}'
        )
        return test_indices, validation_indices, train_indices

    def create_default_folds(self, group_keys: pd.DataFrame) -> None:
        """
        Create default folds based on group keys and save them as CSV files.

        Parameters:
            group_keys (pd.DataFrame): DataFrame containing the group keys.
        """
        all_folds_subjects = []
        all_folds_items = []
        n_splits = self.num_folds

        def _split_and_collect(batch_data: pd.DataFrame, split_ind: int):
            subjects = batch_data[self.subject_column]
            items = batch_data[self.item_columns].astype(str).apply('_'.join, axis=1)
            if self.stratify:
                splitter = StratifiedGroupKFold(n_splits=n_splits)
                y = batch_data[self.stratify]
            else:
                splitter = GroupKFold(n_splits=n_splits)
                y = None

            _, test_subjects_indx = list(
                splitter.split(subjects, y=y, groups=subjects)
            )[split_ind]
            _, test_items_indx = list(splitter.split(items, y=y, groups=items))[
                split_ind
            ]

            return subjects.iloc[test_subjects_indx].tolist(), items.iloc[
                test_items_indx
            ].tolist()

        for split_ind in range(n_splits):
            fold_subjects = []
            fold_items = []

            if self.higher_level_split_column:
                for i_split in group_keys[self.higher_level_split_column].unique():
                    batch_data = group_keys[
                        group_keys[self.higher_level_split_column] == i_split
                    ].reset_index(drop=True)
                    batch_subjects, batch_items = _split_and_collect(
                        batch_data, split_ind
                    )
                    fold_subjects.extend(batch_subjects)
                    fold_items.extend(batch_items)
            else:
                batch_subjects, batch_items = _split_and_collect(
                    group_keys.reset_index(drop=True), split_ind
                )
                fold_subjects.extend(batch_subjects)
                fold_items.extend(batch_items)

            all_folds_subjects.append(fold_subjects)
            all_folds_items.append(fold_items)

        try:
            folds_path = Path(HydraConfig.get().runtime.output_dir) / 'folds'
        except Exception:  # no hydra
            logger.info(
                f'HydraConfig not found. Using default path. Loading folds from {self.folds_path}'
            )
            folds_path = self.folds_path
        subject_folds_path = folds_path / 'subjects'
        item_folds_path = folds_path / 'items'

        for i, (subject_fold, item_fold) in enumerate(
            zip(all_folds_subjects, all_folds_items)
        ):
            item_folds_path.mkdir(parents=True, exist_ok=True)
            subject_folds_path.mkdir(parents=True, exist_ok=True)
            subject_df = pd.DataFrame(sorted(list(set(subject_fold))))
            self.subject_folds[i] = subject_df
            subject_df.to_csv(
                subject_folds_path / f'fold_{i}.csv', header=False, index=False
            )
            item_df = pd.DataFrame(sorted(list(set(item_fold))))
            self.item_folds[i] = item_df
            item_df.to_csv(item_folds_path / f'fold_{i}.csv', header=False, index=False)

    @staticmethod
    def get_combined_indices(fold_dict: dict, folds_indices: list[int]) -> pd.Series:
        """
        Combine fold indices from a fold dictionary based on specified fold indices.

        Parameters:
            fold_dict (dict): Dictionary containing folds.
            folds_indices (list[int]): List of fold indices to combine.

        Returns:
            pd.Series: Combined fold indices as a Series.
        """
        return pd.concat(
            [fold_dict[i] for i in folds_indices], ignore_index=True
        ).squeeze('columns')

    def get_train_val_test_splits(
        self,
        group_keys: pd.DataFrame,
        fold_index: int,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Split the data into training, validation, and test sets.

        Parameters:
            group_keys (pd.DataFrame): DataFrame containing group keys.
            fold_index (int): The fold index to use for splitting.

        Returns:
            tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
            A tuple containing the training keys, a list of validation keys, and a list of test keys.
        """
        test_indices, val_indices, train_indices = self.get_fold_indices(fold_index)
        subject_folds = self.subject_folds
        item_folds = self.item_folds

        # Get subject and item IDs for each split into train/val/test
        subject_train_indices = FoldSplitter.get_combined_indices(
            subject_folds, train_indices
        )
        subject_val_indices = FoldSplitter.get_combined_indices(
            subject_folds, val_indices
        )
        subject_test_indices = FoldSplitter.get_combined_indices(
            subject_folds, test_indices
        )
        item_train_indices = FoldSplitter.get_combined_indices(
            item_folds, train_indices
        )
        item_val_indices = FoldSplitter.get_combined_indices(item_folds, val_indices)
        item_test_indices = FoldSplitter.get_combined_indices(item_folds, test_indices)

        # Get trial-level indices in group_keys per split
        train_subjects_indx = self.get_split_indices(
            group_keys, subject_train_indices, is_item=False
        )
        val_subjects_indx = self.get_split_indices(
            group_keys, subject_val_indices, is_item=False
        )
        test_subjects_indx = self.get_split_indices(
            group_keys, subject_test_indices, is_item=False
        )
        train_items_indx = self.get_split_indices(
            group_keys, item_train_indices, is_item=True
        )
        val_items_indx = self.get_split_indices(
            group_keys, item_val_indices, is_item=True
        )
        test_items_indx = self.get_split_indices(
            group_keys, item_test_indices, is_item=True
        )

        train_indices = np.array(train_subjects_indx.intersection(train_items_indx))

        seen_subject_unseen_item_test_indices = np.array(
            test_items_indx.intersection(train_subjects_indx.union(val_subjects_indx))
        )
        unseen_subject_seen_item_test_indices = np.array(
            train_items_indx.union(val_items_indx).intersection(test_subjects_indx)
        )
        unseen_subject_unseen_item_test_indices = np.array(
            test_items_indx.intersection(test_subjects_indx)
        )

        unseen_subject_unseen_item_val_indices = np.array(
            val_subjects_indx.intersection(val_items_indx)
        )
        unseen_subject_seen_item_val_indices = np.array(
            val_subjects_indx.intersection(train_items_indx)
        )
        seen_subject_unseen_item_val_indices = np.array(
            train_subjects_indx.intersection(val_items_indx)
        )

        assert len(group_keys) == len(train_indices) + len(
            seen_subject_unseen_item_test_indices
        ) + len(unseen_subject_seen_item_test_indices) + len(
            unseen_subject_unseen_item_test_indices
        ) + len(unseen_subject_unseen_item_val_indices) + len(
            unseen_subject_seen_item_val_indices
        ) + len(seen_subject_unseen_item_val_indices), (
            'Data subsets do not sum to all the data'
        )

        self.assert_no_duplicates(train_indices, 'train_indices')
        train_keys = group_keys.iloc[train_indices]
        train_keys.attrs['name'] = SetNames.TRAIN
        train_keys.attrs['set_name'] = SetNames.TRAIN

        test_key_types = [
            (SetNames.SEEN_SUBJECT_UNSEEN_ITEM, seen_subject_unseen_item_test_indices),
            (SetNames.UNSEEN_SUBJECT_SEEN_ITEM, unseen_subject_seen_item_test_indices),
            (
                SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM,
                unseen_subject_unseen_item_test_indices,
            ),
        ]
        test_keys_list = []
        for key_name, indices in test_key_types:
            self.assert_no_duplicates(indices, key_name)
            keys = group_keys.iloc[indices]
            keys.attrs['name'] = key_name
            keys.attrs['set_name'] = SetNames.TEST
            test_keys_list.append(keys)

        val_keys_list = []
        val_key_types = [
            (SetNames.SEEN_SUBJECT_UNSEEN_ITEM, seen_subject_unseen_item_val_indices),
            (SetNames.UNSEEN_SUBJECT_SEEN_ITEM, unseen_subject_seen_item_val_indices),
            (
                SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM,
                unseen_subject_unseen_item_val_indices,
            ),
        ]
        for key_name, indices in val_key_types:
            self.assert_no_duplicates(indices, key_name)
            keys = group_keys.iloc[indices]
            keys.attrs['name'] = key_name
            keys.attrs['set_name'] = SetNames.VAL
            val_keys_list.append(keys)

        self.print_group_info('Train', train_keys)
        for keys in val_keys_list:
            self.print_group_info(f'Val {keys.attrs["name"]}', keys)
        for keys in test_keys_list:
            self.print_group_info(f'Test {keys.attrs["name"]}', keys)

        all_keys = pd.concat([train_keys] + val_keys_list + test_keys_list).sort_index()
        self.print_group_info('All', all_keys)

        return train_keys, val_keys_list, test_keys_list

    @staticmethod
    def print_group_info(name: str, keys: pd.DataFrame) -> None:
        """
        Print group information.

        Parameters:
            name (str): The name of the group.
            keys (pd.DataFrame): DataFrame containing the group keys.
        """
        logger.info(
            f'{name}: # Trials: {len(keys)}. '
            f'# Items: {keys[Fields.UNIQUE_PARAGRAPH_ID].nunique()}; '
            f'# Subjects: {keys[Fields.SUBJECT_ID].nunique()}'
        )

    @staticmethod
    def assert_no_duplicates(indices: list, indices_name: str) -> None:
        """
        Assert that there are no duplicate indices.

        Parameters:
            indices: The indices to check.
            indices_name: Name of the indices for error reporting.

        Raises:
            AssertionError: If duplicates are found in the indices.
        """
        assert len(indices) == len(set(indices)), indices_name + ' contains duplicates'

    def create_trial_folds(
        self, group_keys: pd.DataFrame, eval_regime_names, trial_ids_folder: Path
    ) -> None:
        # Create and save evaluation regimes for each fold
        for fold_index in range(self.num_folds):
            train_keys, val_keys_list, test_keys_list = self.get_train_val_test_splits(
                group_keys=group_keys,
                fold_index=fold_index,
            )
            eval_regimes = [train_keys] + val_keys_list + test_keys_list

            # Save all regimes to a single CSV file for this fold
            regimes_csv_path = (
                trial_ids_folder / f'fold_{fold_index}_trial_ids_by_regime.csv'
            )
            save_eval_regimes_to_csv(eval_regimes, eval_regime_names, regimes_csv_path)
            logger.info(f'Saved evaluation regimes to {regimes_csv_path}')


def split_dataset(
    data_args: DataArgs, recreate_trial_folds: bool, recreate_item_subject_folds: bool
) -> None:
    """
    Splits the dataset into folds and generates reports for different data types.

    This function initializes a FoldSplitter and then produces training, validation,
    and test splits for multiple report types ('ia' and 'fixations').

    Args:
        data_args (DataArgs): Configuration parameters for data splitting.
        recreate_trial_folds (bool): Flag to indicate if trial folds should be recreated.
        recreate_item_subject_folds (bool): Flag to indicate if item/subject folds should be
    """
    folds_folder_path = data_args.base_path / 'folds_metadata'

    splitter = FoldSplitter(
        item_columns=data_args.split_item_columns,
        subject_column=data_args.subject_column,
        groupby_columns=data_args.groupby_columns,
        num_folds=data_args.n_folds,
        stratify=data_args.stratify,
        folds_path=folds_folder_path,
        higher_level_split_column=data_args.higher_level_split,
    )

    eval_regime_names = (
        [f'{SetNames.TRAIN}_{SetNames.TRAIN}']
        + [f'{SetNames.VAL}_{fold_name}' for fold_name in REGIMES]
        + [f'{SetNames.TEST}_{fold_name}' for fold_name in REGIMES]
    )

    # Retrieve data paths from configuration.
    data_args = get_data_args(data_args.dataset_name)
    report_details = [
        (DataType.IA, data_args.ia_path),
        (DataType.FIXATIONS, data_args.fixations_path),
        (DataType.TRIAL_LEVEL, data_args.trial_level_path),
    ]

    for report_type, data_path in report_details:
        split_data_report(
            data_path=data_path,
            report_type=report_type,
            data_config=data_args,
            splitter=splitter,
            eval_regime_names=eval_regime_names,
            folds_folder_path=folds_folder_path,
            recreate_trial_folds=recreate_trial_folds,
            recreate_item_subject_folds=recreate_item_subject_folds,
        )
        # makes sure that we don't recreate the folds again
        if recreate_trial_folds:
            recreate_trial_folds = False
        if recreate_item_subject_folds:
            recreate_item_subject_folds = False


def split_data_report(
    data_path: Path,
    report_type: DataType,
    data_config: DataArgs,
    splitter: FoldSplitter,
    eval_regime_names: list[str],
    folds_folder_path: Path,
    recreate_trial_folds: bool,
    recreate_item_subject_folds: bool,
) -> None:
    """
    Generates and saves data reports for a specified report type.

    The function loads the raw data from a given path, converts grouping columns to strings,
    and organizes the data into folds using the provided splitter. It saves the evaluation
    regimes to a CSV file and uses them to create data subsets which are saved as Feather files.

    Args:
        data_path (Path): Path to the raw data file.
        report_type (DataType): The type of report, e.g., 'ia' or 'fixations'.
        data_config (DataConfig): Data configuration settings.
        splitter (FoldSplitter): Instance responsible for splitting the dataset.
        eval_regime_names (List[str]): Names for each evaluation regime.
        folds_folder_path (Path): Directory where fold files will be stored.
        recreate_trial_folds (bool): Flag to indicate if trial folds should be recreated.
        recreate_item_subject_folds (bool): Flag to indicate if item/subject folds should be recreated.
    """
    # Load raw data and ensure grouping columns are treated as strings.
    df = load_raw_data(data_path)
    if 'RCS_score' in df.columns:
        df['RCS_score'] = df['RCS_score'].fillna(-1)
    grouped_data = df.groupby(data_config.groupby_columns)
    group_keys = pd.DataFrame(
        data=list(grouped_data.groups), columns=data_config.groupby_columns
    )

    # Check if folds should be created or already exist
    trial_ids_folder = folds_folder_path / 'trial_ids'
    trial_ids_folder.mkdir(parents=True, exist_ok=True)

    def check_or_create_folds(
        folder_path: Path, recreate_flag: bool, action: callable
    ) -> None:
        """
        Check if folds exist or need to be recreated, and perform the specified action.

        Args:
            folder_path (Path): Path to the folder containing fold files.
            recreate_flag (bool): Flag indicating whether to recreate the folds.
            action (callable): Function to execute if folds need to be created.
        """
        if recreate_flag or not all(
            (folder_path / f'fold_{fold_index}_trial_ids_by_regime.csv').exists()
            for fold_index in range(data_config.n_folds)
        ):
            action()

    # Step 1: Create or load item/subject folds
    if recreate_item_subject_folds:
        check_or_create_folds(
            trial_ids_folder,
            recreate_item_subject_folds,
            lambda: splitter.create_default_folds(group_keys),
        )
    else:
        splitter.load_folds()

    # Step 2: Create or load trial folds
    check_or_create_folds(
        trial_ids_folder,
        recreate_trial_folds,
        lambda: splitter.create_trial_folds(
            group_keys, eval_regime_names, trial_ids_folder
        ),
    )

    # Step 3: Use the saved evaluation regimes to create data subsets
    for fold_index in range(data_config.n_folds):
        # Define the path for the trial IDs CSV for the current fold
        trial_ids_csv_path = (
            trial_ids_folder / f'fold_{fold_index}_trial_ids_by_regime.csv'
        )

        # Load the evaluation regimes from CSV
        eval_regimes = load_eval_regimes_from_csv(
            trial_ids_csv_path, data_config.groupby_columns
        )

        # Create the data subsets using the loaded regimes
        for regime_data, regime_name in zip(eval_regimes, eval_regime_names):
            subset_df = _get_data(
                raw_data=df,
                groups=grouped_data.groups,
                group_keys=regime_data,
            )

            save_fold_data(
                subset_df=subset_df,
                fold_dir=folds_folder_path.parent / 'folds' / f'fold_{fold_index}',
                report_type=report_type,
                regime_name=regime_name,
            )


def save_eval_regimes_to_csv(
    eval_regimes: list, regime_names: list, save_path: Path
) -> None:
    """
    Saves the evaluation regimes to a single CSV file.

    Args:
        eval_regimes (list): List of dataframes containing the group keys for each regime.
        regime_names (list): Names of the evaluation regimes.
        save_path (Path): Path where to save the CSV file.
    """
    # Create a combined dataframe with a column indicating the regime
    all_regimes_df = pd.DataFrame()

    for regime_df, regime_name in zip(eval_regimes, regime_names):
        regime_df = regime_df.copy()
        regime_df['regime'] = regime_name
        all_regimes_df = pd.concat([all_regimes_df, regime_df])

    # Save to CSV
    all_regimes_df.to_csv(save_path, index=False)


def load_eval_regimes_from_csv(csv_path: Path, groupby_columns: list) -> list:
    """
    Loads evaluation regimes from a CSV file.

    Args:
        csv_path (Path): Path to the CSV file containing the regimes.
        groupby_columns (list): Columns used for grouping.

    Returns:
        list: List of dataframes, each containing the group keys for a regime.
    """
    # Load the combined dataframe
    all_regimes_df = pd.read_csv(csv_path)

    # Split back into separate dataframes based on regime column
    regimes = []
    for regime_name in all_regimes_df['regime'].unique():
        regime_df = all_regimes_df[all_regimes_df['regime'] == regime_name].copy()
        regime_df = regime_df[groupby_columns]  # Keep only the groupby columns
        regimes.append(regime_df)

    return regimes


def save_fold_data(
    subset_df: pd.DataFrame,
    fold_dir: Path,
    report_type: str,
    regime_name: str,
) -> None:
    """
    Saves the subset of data for a specific fold and regime.

    Args:
        subset_df (pd.DataFrame): The subset of data to save.
        fold_dir (Path): Directory for the current fold.
        report_type (str): The type of report, e.g., 'ia' or 'fixations'.
        regime_name (str): The name of the evaluation regime.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_path = fold_dir / f'{report_type}_{regime_name}.feather'
    subset_df.to_feather(save_path)
    logger.info(f'Saved {save_path}')


def _get_data(
    raw_data: pd.DataFrame,
    groups: dict[tuple, pd.Index],
    group_keys: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extracts data from a Pandas DataFrame based on group keys.

    Args:
        raw_data (pd.DataFrame): The original data from which to extract data.
        groups (dict): A dictionary that maps group names to indices.
        group_keys (list): A list of group names to extract data for.

    Returns:
        pd.DataFrame: A new DataFrame containing the extracted data.
    """
    all_indices = []

    for key_ in group_keys.itertuples(name=None, index=False):
        try:
            all_indices.append(groups[key_])
        except:  # noqa: E722
            breakpoint()

    # Concatenate all indices once, avoid repeated union
    combined_index = pd.Index(np.concatenate(all_indices))

    return raw_data.loc[combined_index].copy()


def main() -> None:
    """
    Main function to execute dataset splitting.

    Configures the data settings and initiates the data splitting process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument(
        '--do_not_recreate_trial_folds',
        action='store_true',
        help='Recreate trial folds',
    )
    parser.add_argument(
        '--do_not_recreate_item_subject_folds',
        action='store_true',
        help='Recreate item/subject folds',
    )
    args = parser.parse_args()
    recreate_trial_folds = not args.do_not_recreate_trial_folds
    recreate_item_subject_folds = not args.do_not_recreate_item_subject_folds
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
            logger.info(f'Splitting {dataset_name}...')
            split_dataset(data_args, recreate_trial_folds, recreate_item_subject_folds)
        # except ValueError as e:
        #     logger.info(f'ValueError splitting {dataset_name}: {e}')
        except FileNotFoundError as e:
            logger.warning(f'FileNotFoundError splitting {dataset_name}: {e}')
        # except KeyError as e:
        #     logger.info(f'KeyError splitting {dataset_name}: {e}')


if __name__ == '__main__':
    main()
