from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.configs.constants import (
    BINARY_P_AND_Q_TASKS,
    BINARY_PARAGRAPH_ONLY_TASKS,
    REGRESSION_PARAGRAPH_ONLY_TASKS,
    Fields,
)
from src.configs.main_config import Args
from src.configs.models.base_model import DLModelArgs, MLModelArgs
from src.data.utils import load_raw_data


class TextDataSet(TorchDataset):
    """
    A PyTorch dataset for text data.
    """

    def __init__(self, cfg: Args):
        self.prediction_mode = cfg.data.task
        valid_modes = (
            BINARY_P_AND_Q_TASKS
            + BINARY_PARAGRAPH_ONLY_TASKS
            + REGRESSION_PARAGRAPH_ONLY_TASKS
        )
        if self.prediction_mode not in valid_modes:
            raise ValueError(
                f'Invalid value for PREDICTION_MODE: {self.prediction_mode}'
            )
        self.max_data_seq_len = cfg.data.max_seq_len
        self.max_model_supported_len = cfg.model.max_supported_seq_len
        self.actual_max_needed_len = min(
            self.max_data_seq_len, self.max_model_supported_len
        )
        self.num_special_tokens_to_add = cfg.model.num_special_tokens_add
        self.actual_max_seq_len = 0
        self.max_q_len = cfg.data.max_q_len
        assert isinstance(cfg.model, (DLModelArgs, MLModelArgs))
        self.prepend_eye_features_to_text = cfg.model.prepend_eye_features_to_text
        self.text_key_field = cfg.data.unique_trial_id_column
        self.preorder = cfg.model.preorder

        self.print_tokens = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.backbone,  # type: ignore
            is_split_into_words=True,
            add_prefix_space=True,
        )
        eye_token = '<eye>'
        self.tokenizer.add_special_tokens(
            special_tokens_dict={'additional_special_tokens': [eye_token]},
            replace_additional_special_tokens=False,
        )
        self.eye_token_id: int = self.tokenizer.convert_tokens_to_ids(eye_token)

        text_data = self.prepare_text_data(data_path=cfg.data.ia_path)
        # create a dict mapping from key column (as the dict key) to index (as the dict value)
        text_keys = text_data[self.text_key_field].copy()
        self.key_to_index = dict(zip(text_keys, text_keys.index))

        (
            self.text_features,
            self.inversions_lists,
        ) = self.convert_examples_to_features(
            text_data,
        )

        self.text_data = text_data

    def prepare_text_data(self, data_path: Path) -> pd.DataFrame:
        """
        Prepares the text data by loading it from a CSV file and selecting relevant columns.
        Args:
            data_path (Path): The path to the CSV file containing the text data.

        Returns:
            pd.DataFrame: A DataFrame containing the selected columns from the CSV file
                after dropping duplicates.
        """
        usecols = [
            field.value
            for field in [
                Fields.UNIQUE_TRIAL_ID,
                Fields.QUESTION,
                Fields.PARAGRAPH,
            ]
        ]

        text_data = load_raw_data(data_path)

        missing_columns = [col for col in usecols if col not in text_data.columns]
        if missing_columns:
            logger.warning(f'Missing columns: {missing_columns}')
        existing_columns = [col for col in usecols if col in text_data.columns]
        logger.info(f'Using columns: {existing_columns}')

        text_data = text_data[existing_columns].copy()
        text_data = text_data.drop_duplicates(subset=self.text_key_field).reset_index(
            drop=True
        )
        return text_data

    def __len__(self) -> int:
        return len(self.key_to_index)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], list[int]]:
        features = self.text_features[index]
        inversions_list = self.inversions_lists[index]
        return features, inversions_list

    def convert_examples_to_features(
        self,
        examples: pd.DataFrame,
    ) -> tuple[torch.Tensor | TorchTensorDataset, list[list[int]]]:
        # Roberta tokenization
        """Loads a data file into a list of `InputBatch`s."""

        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None
        paragraphs_input_ids_list = []
        paragraphs_masks_list = []
        input_ids_list: list[list[int] | list[list[int]]] = []
        input_masks_list: list[list[int] | list[list[int]]] = []
        passages_length = []
        inversions_list = []
        full_lengths = []
        for example in tqdm(
            examples.itertuples(),
            total=len(examples),
            desc='Tokenizing',
        ):
            paragraph_ids, inversions, full_length = self.tokenize(
                text=example.paragraph
            )
            full_lengths.append(full_length)
            # TODO Low priority: refactor to avoid duplication of input_ids and p_input_ids
            p_input_ids = paragraph_ids.copy()

            p_input_ids.insert(0, self.tokenizer.cls_token_id)

            # Zero-pad up to the sequence length.
            p_input_mask = [1] * len(p_input_ids) + [0] * (
                self.actual_max_needed_len - len(p_input_ids)
            )
            p_input_ids = p_input_ids + [1] * (
                self.actual_max_needed_len - len(p_input_ids)
            )

            # Add the paragraph to the lists
            paragraphs_input_ids_list.append(p_input_ids)
            paragraphs_masks_list.append(p_input_mask)

            endings_ids = self.add_tokenized_question_if_needed(example)
            full_ending_ids = []
            for ending_tokens in endings_ids:
                full_ending_ids.extend(
                    ending_tokens
                )  # * If adding more than one ending, concatenate them. Consider adding separators.

            input_ids, input_masks = self.process_example(
                paragraph_ids, full_ending_ids
            )
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)

            if self.print_tokens:
                if isinstance(input_ids_list[0][0], list):
                    for ids in input_ids_list[0]:
                        logger.info(self.tokenizer.convert_ids_to_tokens(ids))
                else:
                    logger.info(self.tokenizer.convert_ids_to_tokens(input_ids_list[0]))
                self.print_tokens = False

            passages_length.append(len(paragraph_ids))
            inversions_list.append(inversions)
        if self.actual_max_needed_len > self.actual_max_seq_len:
            logger.warning(
                f'{self.actual_max_needed_len=} while max length in practice is {self.actual_max_seq_len}.'
            )

        features = TorchTensorDataset(
            torch.tensor(paragraphs_input_ids_list, dtype=torch.long),
            torch.tensor(paragraphs_masks_list, dtype=torch.long),
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(input_masks_list, dtype=torch.long),
            torch.tensor(passages_length, dtype=torch.long),
            torch.tensor(full_lengths, dtype=torch.long),
        )

        return features, inversions_list

    def build_inputs_with_special_tokens(
        self,
        context_ids: list[int],
        ending_ids: list[int],
    ) -> list[int]:
        """
        Based on from RobertaTokenizer.build_inputs_with_special_tokens
        #! Check where things break if making changes here
        """
        assert self.tokenizer.cls_token_id is not None
        assert self.tokenizer.sep_token_id is not None

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        input_ids = [cls_token_id]

        if self.prepend_eye_features_to_text:
            input_ids.extend([self.eye_token_id, sep_token_id])

        input_ids += (
            context_ids + [sep_token_id, sep_token_id] + ending_ids + [sep_token_id]
        )
        return input_ids

    def process_example(
        self,
        paragraph_ids: list[int],
        ending_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        input_ids = self.build_inputs_with_special_tokens(paragraph_ids, ending_ids)

        self.verify_input_length(input_ids)
        padding_length = self.actual_max_needed_len - len(input_ids)
        # Update input mask and padding for the concatenated sequence
        input_mask = [1] * len(input_ids) + [0] * padding_length
        padding_ids = [1] * padding_length  # 1 for roberta
        input_ids.extend(padding_ids)

        return input_ids, input_mask

    def add_tokenized_question_if_needed(
        self,
        example,
    ) -> list[list[int]]:
        """
        Processing of example endings based on prediction mode.
        """
        if self.prediction_mode in BINARY_P_AND_Q_TASKS:
            endings = [f'Question: {example.question}']
        else:
            endings = []

        endings_ids: list[list[int]] = [self.tokenize(ending)[0] for ending in endings]

        return endings_ids

    def verify_input_length(self, tokens: list[int]) -> None:
        assert len(tokens) <= self.actual_max_needed_len, (
            f'tokens length is {len(tokens)}, max_seq_length is {self.actual_max_needed_len}'
        )

        if len(tokens) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(tokens)

    def tokenize(self, text: str) -> tuple[list[int], list[int], int]:
        """
        Tokenizes a paragraph into a list of tokens.
        If the tokenized text exceeds actual_max_needed_len, truncates to keep the last actual_max_needed_len tokens.

        Args:
            text (str): The paragraph to tokenize.

        Returns:
            tuple[list[str], list[int]]: The tokenized paragraph and the inversions list.

        """
        tokens = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            add_special_tokens=False,
        )
        input_ids: list[int] = tokens['input_ids']
        token_word_ids: list[int] = tokens.word_ids()
        full_length = max(token_word_ids) + 1
        # Truncate to actual_max_needed_len, keeping the last tokens
        max_length = (
            self.actual_max_needed_len - self.num_special_tokens_to_add - self.max_q_len
        )

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            token_word_ids = token_word_ids[-max_length:]
            min_id = min(token_word_ids)
            token_word_ids = [id_ - min_id for id_ in token_word_ids]

        return input_ids, token_word_ids, full_length
