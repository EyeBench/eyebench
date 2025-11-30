import itertools
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.utils.data.dataset
from loguru import logger
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from src.configs.constants import (
    FEATURES_CACHE_FOLDER,
    SCANPATH_PADDING_VAL,
    DataType,
    DLModelNames,
    Fields,
    NormalizationModes,
    PredMode,
    SetNames,
)
from src.configs.main_config import Args
from src.configs.models.base_model import DLModelArgs, MLModelArgs
from src.data.datasets.TextDataSet import TextDataSet
from src.data.utils import load_fold_data

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # to avoid warnings


class ETDataset(TorchDataset):
    """
    A base class for eye tracking datasets.

    Attributes:
        set_name (SetNames): The name of the set (e.g., train, test, val).
        regime_name (SetNames): The name of the regime (e.g., unseen_subject_seen_item).
    """

    def __init__(
        self,
        cfg: Args,
        set_name: SetNames,
        regime_name: SetNames,
        ia_scaler: MinMaxScaler | RobustScaler | StandardScaler | None = None,
        fixation_scaler: MinMaxScaler | RobustScaler | StandardScaler | None = None,
        trial_features_scaler: MinMaxScaler
        | RobustScaler
        | StandardScaler
        | None = None,
        text_data: TextDataSet | None = None,
    ):
        super().__init__()
        self.set_name = set_name
        self.regime_name = regime_name
        self.ia_scaler = ia_scaler
        self.fixation_scaler = fixation_scaler
        self.trial_features_scaler = trial_features_scaler
        self.use_fixation_data = cfg.model.use_fixation_report
        self.ia_feature_cols = cfg.model.ia_features
        self.fixation_feature_cols = (
            (
                cfg.model.fixation_features
                + cfg.model.ia_features_to_add_to_fixation_data
            )
            if self.use_fixation_data
            else []
        )
        assert isinstance(cfg.model, (DLModelArgs, MLModelArgs))
        self.ia_categorical_features = cfg.model.ia_categorical_features
        self.compute_trial_level_features = cfg.model.compute_trial_level_features
        self.max_data_seq_len = cfg.data.max_seq_len
        self.max_model_supported_len = cfg.model.max_supported_seq_len
        self.actual_max_needed_len = min(
            self.max_data_seq_len, self.max_model_supported_len
        )
        self.max_scanpath_len = cfg.data.max_scanpath_length
        self.max_tokens_in_word = cfg.data.max_tokens_in_word
        self.normalize = cfg.model.normalization_mode
        self.prediction_mode = cfg.data.task
        self.base_model_name = cfg.model.base_model_name
        self.model_name = cfg.model.model_name
        self.prepend_eye_features_to_text = cfg.model.prepend_eye_features_to_text
        self.item_level_features_modes = cfg.model.item_level_features_modes
        self.print_first_nan_occurrences = True
        self.actual_max_tokens_in_word = 0
        self.data_name = cfg.data.dataset_name
        self.folds_folder_name = cfg.data.folds_folder_name
        if text_data is not None:
            self.n_tokens = len(text_data.tokenizer)
            self.eye_token_id = text_data.eye_token_id
            self.sep_token_id = text_data.tokenizer.sep_token_id
        self.target_column = cfg.data.target_column
        self.is_reg = len(list(cfg.data.class_names)) == 1
        self.trial_groupby_columns = cfg.data.groupby_columns

        (
            self.features,
            self.labels,
            self.grouped_ia_data,
            self.grouped_fixation_data,
            self.grouped_raw_fixation_scanpath_ia_labels,
            self.trial_level_features,
            self.trial_level_feature_names,
            self.ordered_key_list,
            self.ia_scaler,
            self.fixation_scaler,
            self.trial_features_scaler,
        ) = ETDataset.cache_or_load_feature(
            cache_file_path=self.create_features_identifier(cfg=cfg),
            overwrite_feature=cfg.trainer.overwrite_data,
            create_feature_func=self.prepare_data,
            create_feature_func_args=dict(
                text_data=text_data,
                cfg=cfg,
            ),
        )

    @staticmethod
    def organize_label_counts(
        labels: list[int], label_names: list[str]
    ) -> pd.DataFrame:
        """
        Organize label counts into a DataFrame.

        Args:
            labels (list): The labels to organize.
            label_names (str): The label names.

        Returns:
            pd.DataFrame: The organized label counts.
        """
        label_counts = np.unique(labels, return_counts=True)
        label_counts = pd.DataFrame(label_counts, index=['label', 'count']).T
        label_counts['percent'] = (
            label_counts['count'] / label_counts['count'].sum() * 100
        )

        label_counts['percent'] = (
            label_counts['percent']
            .astype(
                float,
            )
            .round(2)
        )
        label_counts.attrs['name'] = label_names
        return label_counts

    @staticmethod
    def normalize_features(
        x: pd.DataFrame | pd.Series,
        normalize: NormalizationModes,
        scaler: MinMaxScaler | RobustScaler | StandardScaler,
    ) -> np.ndarray:
        """
        Normalize features based on the specified mode.

        Args:
            x (pd.DataFrame | pd.Series): The features to normalize.
            normalize (NormalizationModes): The normalization mode.
            scaler (MinMaxScaler | RobustScaler | StandardScaler): The scaler to use.
        Returns:
            np.ndarray: The normalized features.
        """

        if normalize == NormalizationModes.NONE:
            return x.to_numpy()
        x_input = pd.DataFrame(x).T if isinstance(x, pd.Series) else x
        if normalize == NormalizationModes.ALL:
            normalized_x = scaler.transform(x_input)
        elif normalize == NormalizationModes.TRIAL:
            normalized_x = scaler.fit_transform(x_input)
        else:
            raise ValueError(
                f'Invalid value for normalize: {normalize}, type: {type(normalize)}',
            )
        return normalized_x

    @staticmethod
    def cache_or_load_feature(
        cache_file_path: Path,
        overwrite_feature: bool,
        create_feature_func: Callable,
        create_feature_func_args: dict[str, Any],
    ) -> tuple:
        """
        Cache or load a feature from disk.

        Args:
            cache_file_path (Path): The path to the cache file.
            overwrite_feature (bool): Whether to overwrite existing feature.
            create_feature_func (Callable): The function to create the feature.
            create_feature_func_args (dict): The arguments for the feature creation function.

        Returns:
            np.ndarray |
            pd.DataFrame |
            torch.utils.data.dataset.TensorDataset |
            tuple[torch.utils.data.dataset.TensorDataset, torch.Tensor]
                The cached or loaded feature.
        """
        if overwrite_feature or not cache_file_path.exists():
            cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Caching features to {cache_file_path}')
            feature = create_feature_func(**create_feature_func_args)
            joblib.dump(feature, cache_file_path, compress=('zlib', 3))
        else:
            logger.info(f'Loading features from {cache_file_path}')
            feature = joblib.load(cache_file_path)
            if type(feature) not in [
                np.ndarray,
                pd.DataFrame,
                torch.utils.data.dataset.TensorDataset,
                tuple,
            ]:
                raise ValueError(
                    'Feature is not a numpy array / pytorch tensor / pandas dataframe',
                )

        return feature

    @staticmethod
    def fit_scaler_if_not_fitted(
        scaler: MinMaxScaler | RobustScaler | StandardScaler,
        raw_data: pd.DataFrame,
        set_name: SetNames,
        feature_columns: list[str] | None = None,
        ia_categorical: list[str] = [],
    ) -> MinMaxScaler | RobustScaler | StandardScaler:
        """
        Fit a scaler if it is not already fitted.

        Args:
            scaler (Union[MinMaxScaler, RobustScaler, StandardScaler]):
                The scaler to fit.
            raw_data (pd.DataFrame): The raw data to fit the scaler on.
            feature_columns (Optional[list[str]], optional): The feature columns to use.
                Defaults to None.

        Returns:
            Union[MinMaxScaler, RobustScaler, StandardScaler]: The fitted scaler.
        """
        try:
            check_is_fitted(scaler)
        except NotFittedError as exc:
            if set_name != SetNames.TRAIN:
                raise ValueError(
                    f"Scaler {scaler} is not fitted and set_name is not 'train'.",
                ) from exc
            # TODO Move feature selection out of this function
            if not feature_columns:
                feature_columns = raw_data.columns.to_list()

            numeric_only_df = raw_data[feature_columns].drop(
                columns=ia_categorical,
                errors='ignore',
            )
            non_numeric = numeric_only_df.select_dtypes(
                exclude=['number', 'bool'],
            )
            if not non_numeric.empty:
                raise ValueError(
                    f'Non-numeric columns found in {set_name} set: {non_numeric.columns}',
                ) from exc

            scaler.fit(numeric_only_df)
            logger.info(f'Fitted {scaler} on {numeric_only_df.columns}')
        return scaler

    def __len__(self) -> int:
        """
        Get the number of unique groups in the dataset.

        Returns:
            int: The number of unique groups in the dataset.
        """
        return len(self.grouped_ia_data.groups)

    def __getitem__(
        self,
        idx: int | np.integer,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, tuple, list[str]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the features, labels,
                ordered key list, and trial groupby columns.
        """
        # TODO I think torch dataset is faster and takes less storage than this.
        # Find a way to use it while keeping the names. Maybe store the names in
        # a list as they do not change.
        example_feats = {name: tensor[idx] for name, tensor in self.features.items()}

        return (
            example_feats,
            self.labels[idx],
            self.ordered_key_list[idx],
            self.trial_groupby_columns,
        )

    def convert_examples_to_features(
        self,
        text_data: TextDataSet | None,
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Convert the examples in the dataset to features.

        Args:
            text_data (TextDataSet | None): The text data.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the converted features.
        """

        features = {}

        if self.compute_trial_level_features:
            features.update(self.extract_trial_level_features())

        if self.use_fixation_data:
            features.update(self.get_fixation_features(text_data=text_data))

        if self.ia_feature_cols:
            features.update(self.get_ia_features(text_data=text_data))

        if text_data:
            features.update(self.get_text_features(text_data, features))

        labels = self.get_labels()

        return features, labels

    def get_ia_features(self, text_data: TextDataSet | None) -> dict[str, torch.Tensor]:
        """
        Generate a list of normalized eye data for all trials.

        Returns:
        list[np.ndarray]: A list of normalized eye data.
        """
        eyes_list = []
        for grouped_data_key in tqdm(self.ordered_key_list, desc='IA features'):
            trial = self.grouped_ia_data.get_group(grouped_data_key)
            eyes, _, _ = self.get_eye_data(trial=trial, text_data=text_data)
            eyes_list.append(eyes)

        return {'eyes': torch.tensor(np.array(eyes_list), dtype=torch.float32)}

    def group_to_length(
        self,
        lst: list[int],
        col_pad_to_len: int,
        row_pad_to_len: int,
        inv_list_to_token_word_attn_mask: bool = False,
    ) -> torch.Tensor:
        """
        Pad a list of values to a predefined length.

        Example: [1, 1, 1, 2, 2, 3, 3, 3, 3] ->
            [tensor([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]])]
        Three words, first word has 3 tokens, second word has 2 tokens, third word has 4 tokens.
        Input list assumed to be sorted.
        Used to represent token to word mapping.
        I.e., in input, each token (index in lst) is mapped to a word index (value in lst),
        in output each word index (row) is mapped to a token index (values in row).

        Args:
            lst (list): The list of values to pad.
            col_pad_to_len (int): The length to pad to number of cols.
            row_pad_to_len (int): The length to pad to number of rows.
            inv_list_to_token_word_attn_mask (bool, optional):
                Whether to convert the list to a token-word attention mask. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the padded values.
        """
        # Group the list by the values, and convert to a tensor
        # Example: [1, 1, 1, 2, 2, 3, 3, 3, 3] ->
        #  [tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6, 7, 8])
        grouped_lst = [
            torch.tensor(data=list(group))
            for _, group in itertools.groupby(
                iterable=range(len(lst)),
                key=lambda x: lst[x],
            )
        ]

        if inv_list_to_token_word_attn_mask:
            # [1, 1, 1, 2, 2, 3, 3, 3, 3] -> [tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6, 7, 8])
            #     Before attending previous and next word:
            #     [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            #      [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            #      [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #      ...
            #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

            #      After attending previous and next word:
            #     [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            #      [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #      ...
            #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            # Note: the first column is used to attend to the [CLS] token.

            size = (
                self.actual_max_needed_len
            )  # ! Can be reduced to maximal num of *words* in a paragraph (not tokens)
            matrix = torch.zeros(size + 1, size)
            for i, row in enumerate(grouped_lst):
                matrix[i, 0] = 1
                matrix[i, row + 1] = 1
                # Add attention to the previous and next word
                if i > 0:
                    matrix[i, grouped_lst[i - 1] + 1] = 1
                if i < len(grouped_lst) - 1:
                    matrix[i, grouped_lst[i + 1] + 1] = 1
            return matrix

        current_max_tokens_in_word = max(len(group) for group in grouped_lst)
        if current_max_tokens_in_word > self.actual_max_tokens_in_word:
            self.actual_max_tokens_in_word: int = current_max_tokens_in_word

        # Add padding
        # Example:
        # [
        #   tensor([0, 1, 2]),    -> tensor([0, 1, 2, -2])
        #   tensor([3, 4]),        -> tensor([3, 4, -1, -2])
        #   tensor([5, 6, 7, 8]),  -> tensor([5, 6, 7, 8])
        # ]
        padded_tensor = pad_sequence(
            sequences=grouped_lst,
            batch_first=True,
            padding_value=-2,
        )

        num_padding_cols = max(0, col_pad_to_len - padded_tensor.size(dim=1))
        padding = torch.full(
            size=(padded_tensor.size(dim=0), num_padding_cols),
            fill_value=-2,
        )
        padded_tensor = torch.cat(tensors=(padded_tensor, padding), dim=1)

        # Calculate the number of rows needed to reach the predefined length
        num_padding_rows = max(0, row_pad_to_len - padded_tensor.size(dim=0))

        # Create a tensor of padding values
        padding = torch.full(
            size=(num_padding_rows, padded_tensor.size(dim=1)),
            fill_value=-2,
        )

        # Concatenate the padding to the padded_tensor
        padded_tensor = torch.cat(tensors=(padded_tensor, padding), dim=0)
        padded_tensor += 1
        return padded_tensor

    def create_features_identifier(
        self,
        cfg: Args,
    ) -> Path:
        """
        Create an identifier for features.

        Args:
            cfg (Args): The configuration.

        Returns:
            str: The features identifier.
        """

        return (
            FEATURES_CACHE_FOLDER
            / (f'{self.data_name}_{self.prediction_mode}_{self.model_name}')
            / f'fold_{cfg.data.fold_index}'
            / f'{self.regime_name}_{self.set_name}.pkl'
        )

    def get_trial_text_data(self, text_data: TextDataSet, trial_info: pd.Series):
        """
        Get the text data for a trial.

        Args:
            text_data (TextDataSet): The text data.
            key (str): The key for the trial.
        Returns:
            tuple: The text data for the trial.
        """

        key_ = trial_info[text_data.text_key_field]
        text_index = text_data.key_to_index[key_]
        (
            (
                p_input_ids,
                p_input_masks,
                input_ids,
                input_mask,
                passage_length,
                full_length,
            ),
            inversions_list,
        ) = text_data[text_index]

        return (
            p_input_ids,
            p_input_masks,
            input_ids,
            input_mask,
            passage_length,
            full_length,
            inversions_list,
        )

    def get_text_features(
        self, text_data: TextDataSet, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        input_ids_list = []
        input_masks_list = []
        p_input_ids_list = []
        grouped_inversions = []
        eyes_list = []
        for idx in tqdm(range(len(self.ordered_key_list)), desc='Text features'):
            grouped_data_key = self.ordered_key_list[idx]
            trial = self.grouped_ia_data.get_group(grouped_data_key)
            (
                p_input_ids,
                _,
                input_ids,
                input_mask,
                _,
                _,
                inversions_list,
            ) = self.get_trial_text_data(text_data=text_data, trial_info=trial.iloc[0])

            input_ids_list.append(input_ids)
            input_ids_unsqueeze = input_ids
            input_mask_unsqueeze = input_mask
            if len(input_ids.shape) == 1:
                input_ids_unsqueeze = input_ids_unsqueeze.unsqueeze(dim=0)
                input_mask_unsqueeze = input_mask_unsqueeze.unsqueeze(dim=0)

            p_input_ids_list.append(p_input_ids)

            eyes, eye_seq_len, pad_length = self.get_eye_data(
                trial=trial, text_data=text_data, inversions_list=inversions_list
            )  # TODO recomputes eyes

            inv_list_to_token_word_attn_mask = (
                self.base_model_name == DLModelNames.POSTFUSION_MODEL
            )

            group_inversions = self.group_to_length(
                lst=inversions_list,
                col_pad_to_len=self.max_tokens_in_word,
                row_pad_to_len=self.actual_max_needed_len,
                inv_list_to_token_word_attn_mask=inv_list_to_token_word_attn_mask,
            )
            if self.use_fixation_data:
                scanpath = features['scanpath'][idx, :]
                actual_scanpath = scanpath[scanpath >= 0]  # Remove padding values

                group_inversions = group_inversions[actual_scanpath]

                # Pad group_inversions to max_scanpath_len
                num_padding_rows = self.max_scanpath_len - group_inversions.size(dim=0)
                if num_padding_rows > 0:
                    padding = torch.full(
                        size=(num_padding_rows, group_inversions.size(dim=1)),
                        fill_value=-1,
                    )
                    group_inversions = torch.cat(
                        tensors=(group_inversions, padding), dim=0
                    )

            eyes_list.append(eyes)

            if self.prepend_eye_features_to_text:
                if self.use_fixation_data:
                    pad_len = features['fixation_pads'][idx]
                    seq_len = self.max_scanpath_len - pad_len

                else:
                    seq_len = eye_seq_len
                    pad_len = pad_length

                ones = np.ones(seq_len)
                zeroes = np.zeros(pad_len)
                eye_mask = np.concatenate((ones, zeroes), axis=0)

                axis = 0
                input_mask = torch.from_numpy(
                    np.concatenate((eye_mask, input_mask), axis=axis),
                )

            input_masks_list.append(input_mask)
            grouped_inversions.append(group_inversions)

        if self.max_tokens_in_word > self.actual_max_tokens_in_word:
            logger.warning(
                f'{self.actual_max_tokens_in_word=} but using {self.max_tokens_in_word=}'
            )
        result = {
            'input_ids': torch.stack(input_ids_list),
            'input_masks': torch.stack(input_masks_list),
            'grouped_inversions': torch.stack(grouped_inversions),
            'p_input_ids': torch.stack(p_input_ids_list),
            'eyes': torch.tensor(np.array(eyes_list), dtype=torch.float32),
        }

        return result

    def get_eye_data(
        self, trial: pd.DataFrame, text_data: TextDataSet | None, inversions_list=None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Extract and normalize eye data from a trial.

        Args:
        trial (pd.DataFrame): The trial data.
        text_data (TextDataSet): The text data.
        inversions_list (list, optional): The list of inversions. Defaults to None.

        Returns:
        np.ndarray: The normalized eye data.
        """
        if text_data:
            (
                _,
                _,
                _,
                _,
                _,
                _,
                inversions_list,
            ) = self.get_trial_text_data(text_data=text_data, trial_info=trial.iloc[0])
            length_in_words = max(inversions_list) + 1
            trial = trial.tail(length_in_words).copy()

        eyes = trial[self.ia_feature_cols].drop(
            columns=self.ia_categorical_features,
            errors='ignore',
        )

        eyes = ETDataset.normalize_features(
            eyes,
            normalize=self.normalize,
            scaler=self.ia_scaler,
        )
        num_pre_eye_tokens = 0  # TODO hardcoded value
        if not self.prepend_eye_features_to_text and inversions_list:
            aligned_eyes = [eyes[inv_idx, :] for inv_idx in inversions_list]
            eyes = np.stack(aligned_eyes)
            num_pre_eye_tokens = 1

        eye_seq_len, eyes_dim = eyes.shape
        eyes_pad_left = np.zeros((num_pre_eye_tokens, eyes_dim))
        pad_length = self.actual_max_needed_len - eye_seq_len - num_pre_eye_tokens
        if pad_length < 0:
            logger.error(
                f'Eye data length {eye_seq_len} exceeds max eye length {self.actual_max_needed_len}'
            )
        eyes_pad_right = np.zeros((pad_length, eyes_dim))
        eyes = np.concatenate((eyes_pad_left, eyes, eyes_pad_right))
        eyes = np.nan_to_num(eyes, nan=0.0)  # TODO this shouldn't be needed
        return eyes, eye_seq_len, pad_length

    def get_labels(self) -> torch.Tensor:
        labels_list = []
        for grouped_data_key in tqdm(self.ordered_key_list, desc='Label'):
            trial = self.grouped_ia_data.get_group(grouped_data_key)
            assert trial[self.target_column].nunique() == 1, (
                f'Label {self.target_column} is not the same for all rows in {grouped_data_key}'
            )
            y = trial.iloc[0][self.target_column]

            labels_list.append(y)
        return torch.tensor(
            labels_list, dtype=torch.float32 if self.is_reg else torch.long
        )

    def prepare_data(
        self,
        cfg: Args,
        text_data: TextDataSet | None,
    ) -> tuple:
        # Define a partial function for loading dataframes
        load_data_partial = partial(
            load_fold_data,
            fold_index=cfg.data.fold_index,
            base_path=cfg.data.base_path,
            folds_folder_name=cfg.data.folds_folder_name,
            set_name=self.set_name,
            regime_name=self.regime_name,
        )

        ia_data = load_data_partial(data_type=DataType.IA)
        if cfg.data.task != PredMode.RC and cfg.data.n_questions_per_item > 1:
            before = len(ia_data)
            ia_data = (
                ia_data[ia_data['question_index'].isin([1, 'tq_1'])]
                .drop(columns=['question_index'])
                .copy()
            )
            logger.info(
                f'Kept {len(ia_data)} out of {before} ({(len(ia_data) / before) * 100})% in ia_data'
            )
        filtered_ia = ia_data[
            list(set(self.trial_groupby_columns + self.ia_feature_cols))
        ].copy()
        if filtered_ia.columns[filtered_ia.isna().any()].tolist():
            warnings.warn(
                f'{
                    filtered_ia.columns[filtered_ia.isna().any()].tolist()
                }. Forward filling and backward filling.',
            )
        filtered_ia = filtered_ia.ffill().bfill()
        self.grouped_ia_data = filtered_ia.groupby(self.trial_groupby_columns)
        self.ordered_key_list = list(self.grouped_ia_data.groups)
        if self.ia_feature_cols:
            # filtered_ia = remove_nan_values(filtered_ia)
            self.ia_scaler = self.fit_scaler_if_not_fitted(
                scaler=self.ia_scaler,
                raw_data=filtered_ia,
                set_name=self.set_name,
                feature_columns=self.ia_feature_cols,
                ia_categorical=self.ia_categorical_features,
            )
        else:
            self.ia_scaler = None

        if self.use_fixation_data:
            fixation_data = load_data_partial(data_type=DataType.FIXATIONS)
            if cfg.data.task != PredMode.RC and cfg.data.n_questions_per_item > 1:
                before = len(fixation_data)
                fixation_data = (
                    fixation_data[fixation_data['question_index'].isin([1, 'tq_1'])]
                    .drop(columns=['question_index'])
                    .copy()
                )
                logger.info(f'Removed {len(fixation_data) / before} % duplicate rows')
            filtered_fixations = fixation_data[
                list(
                    set(
                        self.trial_groupby_columns
                        + self.fixation_feature_cols
                        + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
                    )
                )
            ].copy()

            if filtered_fixations.columns[filtered_fixations.isna().any()].tolist():
                warnings.warn(
                    f'{
                        filtered_fixations.columns[
                            filtered_fixations.isna().any()
                        ].tolist()
                    }. Forward filling and backward filling.',
                )
            filtered_fixations = filtered_fixations.ffill().bfill()

            # filtered_fixations = remove_nan_values(filtered_fixations)

            self.grouped_fixation_data = filtered_fixations[
                self.trial_groupby_columns + self.fixation_feature_cols
            ].groupby(
                self.trial_groupby_columns
            )  # TODO add a check that fixation, ia and trial keys are the same
            raw_fixation_scanpath_ia_labels = filtered_fixations[
                self.trial_groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
            ]
            self.grouped_raw_fixation_scanpath_ia_labels = (
                raw_fixation_scanpath_ia_labels.groupby(self.trial_groupby_columns)
            )
            self.fixation_scaler = self.fit_scaler_if_not_fitted(
                scaler=self.fixation_scaler,
                raw_data=filtered_fixations,
                set_name=self.set_name,
                feature_columns=self.fixation_feature_cols,
                ia_categorical=self.ia_categorical_features,
            )
        else:
            self.grouped_fixation_data = None
            self.grouped_raw_fixation_scanpath_ia_labels = None
            self.fixation_scaler = None

        if cfg.model.compute_trial_level_features:
            trial_level_data = load_data_partial(data_type=DataType.TRIAL_LEVEL)
            assert trial_level_data is not None
            ia_feature_names = pd.read_csv(
                cfg.data.processed_data_path / 'ia_trial_level_feature_keys.csv'
            )
            fixation_feature_names = pd.read_csv(
                cfg.data.processed_data_path / 'fixation_trial_level_feature_keys.csv'
            )
            feature_names = pd.concat(
                [ia_feature_names, fixation_feature_names],
                axis=0,
            )
            self.trial_level_feature_names = (
                feature_names[
                    feature_names['feature_type'].isin(self.item_level_features_modes)
                ]['feature_name']
                .drop_duplicates()
                .tolist()
            )
            logger.info(
                f'Using {len(self.trial_level_feature_names)} trial level features.'
            )
            trial_level_data = trial_level_data[self.trial_level_feature_names]
            # keep only trials whose unique_trial_id ends with '1'
            if cfg.data.task != PredMode.RC and cfg.data.n_questions_per_item > 1:
                unique_ids = trial_level_data.index.get_level_values(
                    level='unique_trial_id'
                ).astype(str)
                mask = unique_ids.str.endswith('1')
                before = len(trial_level_data)
                trial_level_data = trial_level_data[mask].copy()
                logger.info(
                    f'Removed {len(trial_level_data) / before} % duplicate rows in trial_level_data'
                )
            self.trial_level_features = trial_level_data
            self.trial_features_scaler = self.fit_scaler_if_not_fitted(
                scaler=self.trial_features_scaler,
                raw_data=self.trial_level_features,
                feature_columns=self.trial_level_feature_names,
                set_name=self.set_name,
                ia_categorical=self.ia_categorical_features,
            )
        else:
            self.trial_level_features = None
            self.trial_level_feature_names = None
            self.trial_features_scaler = None

        features, labels = self.convert_examples_to_features(text_data)

        return (
            features,
            labels,
            self.grouped_ia_data,
            self.grouped_fixation_data,
            self.grouped_raw_fixation_scanpath_ia_labels,
            self.trial_level_features,
            self.trial_level_feature_names,
            self.ordered_key_list,
            self.ia_scaler,
            self.fixation_scaler,
            self.trial_features_scaler,
        )

    def get_fixation_features(
        self, text_data: TextDataSet | None
    ) -> dict[str, torch.Tensor]:
        """
        Convert the examples in the dataset to fixation features.

        Returns:
            tuple: A tuple containing the fixation features, pads, scanpath, and scanpath pads.
        """
        fixation_list = []
        pads_list = []
        scanpath_list = []
        scanpath_pads_list = []
        for grouped_data_key in tqdm(self.ordered_key_list, desc='Fixation features'):
            # Get the data group associated with the given index.
            trial = self.grouped_fixation_data.get_group(
                grouped_data_key,
            ).copy()

            scanpath = self.grouped_raw_fixation_scanpath_ia_labels[
                Fields.FIXATION_REPORT_IA_ID_COL_NAME
            ].get_group(grouped_data_key)
            if text_data:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    full_length,
                    inversions_list,
                ) = self.get_trial_text_data(
                    text_data=text_data,
                    trial_info=trial.iloc[0],
                )
                truncated_words = full_length - (max(inversions_list) + 1)
                trial = trial[
                    (
                        trial[Fields.FIXATION_REPORT_IA_ID_COL_NAME]
                        > int(truncated_words)
                    )
                    | (trial[Fields.FIXATION_REPORT_IA_ID_COL_NAME] == -1)
                ].copy()
                scanpath = scanpath[
                    (scanpath > int(truncated_words)) | (scanpath == -1)
                ].copy()
                # decrease by truncated_words for all that are not -1
                scanpath = scanpath.apply(
                    lambda x: x - int(truncated_words) if x != -1 else x
                )
                for col in (
                    Fields.FIXATION_REPORT_IA_ID_COL_NAME,
                    'NEXT_FIX_INTEREST_AREA_INDEX',
                ):
                    if col in trial.columns:
                        trial[col] = trial[col].apply(
                            lambda x: x - int(truncated_words) if x != -1 else x
                        )

            fixation = trial[self.fixation_feature_cols].drop(
                columns=self.ia_categorical_features,
                errors='ignore',
            )

            fixation = ETDataset.normalize_features(
                fixation,
                normalize=self.normalize,
                scaler=self.fixation_scaler,
            )

            if self.compute_trial_level_features:
                # concat back the "is_content_word" and "ptb_pos" columns from trial
                # TODO BEyeLSTM specific code, save as different variable?
                fixation = np.concatenate(
                    (
                        fixation,
                        trial[['is_content_word', 'ptb_pos']].to_numpy(),
                    ),  # ! Order matters here!
                    axis=1,
                )

            pad_length = self.max_scanpath_len - len(fixation)
            fixation_dim = fixation.shape[1]

            fixation_padding = np.zeros((pad_length, fixation_dim))
            fixation = np.concatenate((fixation, fixation_padding))
            # fixation = fixation[: self.max_scanpath_len] # TODO Do we want this here? Was for PoTeC only
            # pad the scanpath with -1
            scanpath_padding = np.full(pad_length, SCANPATH_PADDING_VAL)
            scanpath = np.concatenate((scanpath, scanpath_padding))
            # scanpath = scanpath[: self.max_scanpath_len] # TODO Do we want this here? Was for PoTeC only

            fixation_list.append(fixation)
            pads_list.append(pad_length)
            scanpath_list.append(scanpath)
            scanpath_pads_list.append(pad_length)

        # TODO sure we want to fillna here?
        fixation_list = [pd.DataFrame(fix_list).fillna(0) for fix_list in fixation_list]
        ret = {
            'fixation_features': torch.tensor(
                np.array(fixation_list).astype(float), dtype=torch.float32
            ),
            'fixation_pads': torch.tensor(pads_list, dtype=torch.long),
            'scanpath': torch.tensor(np.array(scanpath_list), dtype=torch.long),
            'scanpath_pads': torch.tensor(scanpath_pads_list, dtype=torch.long),
        }
        return ret

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
            trial_features = trial_level_features.loc[grouped_data_key]

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
