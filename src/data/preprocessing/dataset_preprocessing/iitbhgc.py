from __future__ import annotations

from ast import literal_eval

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import (
    get_surp_extractor,
)
from text_metrics.surprisal_extractors.extractors_constants import (
    SurpExtractorType,
)
from tqdm import tqdm

from src.configs.constants import DataType, Fields
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.data.utils import (
    add_missing_features,
    compute_trial_level_features,
)

tqdm.pandas()
logger.add('logs/preprocessing.log', level='INFO')


class IITBHGCProcessor(DatasetProcessor):
    """Processor for IITBHGC dataset"""

    # Text fixes mapping as class constant for better performance
    TEXT_FIXES = {
        3: [('with Andy', 'with__NBWS__Andy')],
        5: [('watch Jose', 'watch__NBWS__Jose')],
        9: [('$1,750', '$__NBWS__1,750.00')],
        25: [('£3', '£__NBWS__3.00')],
        26: [('for Virgil', 'for__NBWS__Virgil')],
        33: [('at FC', 'at__NBWS__FC')],
        53: [('$5,000', '$__NBWS__5,000.00')],
        74: [('$20,000', '$__NBWS__20,000.00')],
        82: [('£750,000', '£__NBWS__7,50,000.00')],
        99: [('$5.3', '$__NBWS__5.30')],
        102: [('$10', '$__NBWS__10.00'), ('$9', '$__NBWS__9.00')],
        130: [('$50,000', '$__NBWS__50,000.00')],
        257: [('$10', '$__NBWS__10.00')],
        280: [('Claim: Mile Jedinak twisted', 'Claim: Mile Jedinak__NBWS__twisted')],
        288: [('$2', '$__NBWS__2.00')],
        298: [('$200', '$__NBWS__200.00')],
        325: [('$9', '$__NBWS__9.00')],
        357: [('$1.8', '$__NBWS__1.80'), ('$2.6', '$__NBWS__2.60')],
        365: [('$25,000', '$__NBWS__25,000.00')],
        373: [
            ('$260', '$__NBWS__260.00'),
            ('$1.7', '$__NBWS__1.70'),
            ('$1.37', '$__NBWS__1.37'),
        ],
        403: [('hour-long', 'hour-long hour-long')],
        404: [('$2.9', '$__NBWS__2.90')],
        425: [
            ('- many', '-__NBWS__many'),
            ('against Leicester City', 'against__NBWS__Leicester__NBWS__City'),
            ("mock Louis van Gaal's", "mock Louis van__NBWS__Gaal's"),
        ],
        441: [('$10,000', '$__NBWS__10,000.00')],
        460: [('$1.6', '$__NBWS__1.60')],
        468: [('$105', '$__NBWS__105.00')],
        483: [('(£4,943)', '(£__NBWS__4,943.00)')],
        485: [('long-running', 'long-running long-running')],
    }

    @staticmethod
    def fix_texts(text: str, text_name: np.int64) -> str:
        """Apply text-specific fixes based on text_name."""
        if text_name in IITBHGCProcessor.TEXT_FIXES:
            for old, new in IITBHGCProcessor.TEXT_FIXES[text_name]:
                text = text.replace(old, new)
        return text

    @staticmethod
    def init_word_dict(text_strs: list[str], text_aois: list[int]) -> dict:
        """Initialize word dictionary with default values."""
        return {
            int(word_index): {
                'IA_LABEL': word.replace('__NBWS__', '\xa0'),
                'IA_ID': word_index,
                **{
                    key: 0
                    for key in [
                        'FFD',
                        'SFD',
                        'FD',
                        'FPRT',
                        'FRT',
                        'TFT',
                        'RRT',
                        'RPD_inc',
                        'RPD_exc',
                        'RBRT',
                        'Fix',
                        'FPF',
                        'RR',
                        'FPReg',
                        'TRC_out',
                        'TRC_in',
                        'SL_in',
                        'SL_out',
                        'TFC',
                    ]
                },
            }
            for word_index, word in zip(text_aois, text_strs)
        }

    def compute_word_level_reading_measures(
        self,
        fix_df: pd.DataFrame,
        stim_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute word-level reading measures from fixation data."""

        def process_participant(
            text_name: np.int64, participant_id: str
        ) -> pd.DataFrame:
            try:
                # Get relevant data slices
                aoi_df = stim_df[stim_df[Fields.UNIQUE_PARAGRAPH_ID] == text_name]
                tmp_df = fix_df[fix_df[Fields.UNIQUE_PARAGRAPH_ID] == text_name]
                fixations_df = tmp_df[
                    tmp_df[Fields.SUBJECT_ID] == participant_id
                ].copy()

                if fixations_df.empty:
                    return pd.DataFrame()

                assert len(fixations_df.label.unique()) == 1
                label = fixations_df['label'].iloc[0]

                # Add dummy row
                fixations_df = pd.concat(
                    [
                        fixations_df,
                        pd.DataFrame(
                            [[0] * len(fixations_df.columns)],
                            columns=fixations_df.columns,
                        ),
                    ],
                    ignore_index=True,
                )

                # Process text
                text = self.fix_texts(aoi_df.paragraph.iloc[0], text_name)
                text_strs = text.split()
                text_aois = list(range(len(text_strs)))

                # Initialize word dictionary
                word_dict = self.init_word_dict(text_strs, text_aois)

                # Process fixations
                right_most_word = cur_fix_word_idx = next_fix_word_idx = (
                    next_fix_dur
                ) = -1

                for _, fixation in fixations_df.iterrows():
                    # Skip invalid fixations
                    try:
                        aoi = int(fixation['CURRENT_FIX_X'])
                    except (ValueError, TypeError):
                        continue

                    if (
                        fixation['IA_LABEL'] == '.'
                        or fixation['CURRENT_FIX_DURATION'] == 0
                    ):
                        continue

                    # Update fixation indices
                    last_fix_word_idx = cur_fix_word_idx
                    cur_fix_word_idx = next_fix_word_idx
                    cur_fix_dur = next_fix_dur
                    next_fix_word_idx = aoi
                    next_fix_dur = fixation['CURRENT_FIX_DURATION']

                    # Validate word match
                    if aoi in word_dict:
                        self._validate_word_match(
                            word_dict[aoi]['IA_LABEL'],
                            fixation['IA_LABEL'],
                            text_name,
                            participant_id,
                            aoi,
                        )
                    else:
                        continue

                    if next_fix_dur == 0:
                        next_fix_word_idx = cur_fix_word_idx

                    if cur_fix_word_idx == -1:
                        continue

                    right_most_word = max(right_most_word, cur_fix_word_idx)

                    # Update word statistics
                    cur_word = word_dict[cur_fix_word_idx]
                    cur_word['TFT'] += int(cur_fix_dur)
                    cur_word['TFC'] += 1

                    if cur_word['FD'] == 0:
                        cur_word['FD'] = int(cur_fix_dur)

                    if right_most_word == cur_fix_word_idx:
                        if cur_word['TRC_out'] == 0:
                            cur_word['FPRT'] += int(cur_fix_dur)
                            if last_fix_word_idx < cur_fix_word_idx:
                                cur_word['FFD'] += int(cur_fix_dur)
                    else:
                        word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

                    if cur_fix_word_idx < last_fix_word_idx:
                        cur_word['TRC_in'] += 1
                    if cur_fix_word_idx > next_fix_word_idx:
                        cur_word['TRC_out'] += 1
                    if cur_fix_word_idx == right_most_word:
                        cur_word['RBRT'] += int(cur_fix_dur)

                    if cur_word['FRT'] == 0 and (
                        cur_fix_word_idx != next_fix_word_idx or next_fix_dur == 0
                    ):
                        cur_word['FRT'] = cur_word['TFT']

                    if cur_word['SL_in'] == 0:
                        cur_word['SL_in'] = cur_fix_word_idx - last_fix_word_idx
                    if cur_word['SL_out'] == 0:
                        cur_word['SL_out'] = next_fix_word_idx - cur_fix_word_idx

                # Finalize word measures
                rows = []
                for word_index, word_rm in word_dict.items():
                    if word_rm['FFD'] == word_rm['FPRT']:
                        word_rm['SFD'] = word_rm['FFD']
                    word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
                    word_rm['FPF'] = int(word_rm['FFD'] > 0)
                    word_rm['RR'] = int(word_rm['RRT'] > 0)
                    word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
                    word_rm['Fix'] = int(word_rm['TFT'] > 0)
                    word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']
                    word_rm[Fields.SUBJECT_ID] = participant_id
                    word_rm[Fields.UNIQUE_PARAGRAPH_ID] = text_name
                    word_rm['paragraph'] = text
                    word_rm['word_index'] = word_index
                    word_rm['label'] = label
                    rows.append(word_rm)

                return pd.DataFrame(rows)

            except Exception as e:
                logger.exception(
                    f'Error processing {text_name} - {participant_id}: {e}'
                )
                return pd.DataFrame()

        # Process all participants in parallel-ready structure
        rm_df_parts = [
            process_participant(text_name, participant_id)
            for text_name in fix_df[Fields.UNIQUE_PARAGRAPH_ID].unique()
            for participant_id in fix_df[
                fix_df[Fields.UNIQUE_PARAGRAPH_ID] == text_name
            ][Fields.SUBJECT_ID].unique()
        ]

        return pd.concat(rm_df_parts, ignore_index=True)

    def _validate_word_match(
        self,
        expected: str,
        actual: str,
        text_name: np.int64,
        participant_id: str,
        aoi: int,
    ) -> None:
        """Validate word match between expected and actual labels."""
        if expected == actual:
            return

        # Check for known acceptable mismatches
        if (
            actual in expected
            or expected.strip('(').strip(')')
            in actual  # problems in their aois vs. paragraph
            or expected.lower() == actual.lower()  # case differences
            or text_name
            in [122, 305]  # known issues in these texts [wrong kommata in numbers]
        ):
            logger.info(f'Acceptable mismatch: "{expected}" vs "{actual}"')
            return

        if '\xa0' in actual:
            logger.warning(
                f'Non-breaking space in fixation: text={text_name}, participant={participant_id}, aoi={aoi}'
            )

        logger.warning(
            f'Mismatch in text {text_name} for participant {participant_id} at AOI {aoi}: '
            f'expected "{expected}", got "{actual}"'
        )

    def get_column_map(self, data_type: DataType) -> dict:
        """Get column mapping for IITBHGC dataset."""
        base_map = {
            'trial_id': Fields.UNIQUE_PARAGRAPH_ID,
            'participant_id': Fields.SUBJECT_ID,
            'fixation_word_ids': 'CURRENT_FIX_X',
            'fixation_durations': 'CURRENT_FIX_DURATION',
            'text': 'paragraph',
            'fixation_seqs': 'CURRENT_FIX_INDEX',
            'fixation_word_texts': 'IA_LABEL',
        }
        return base_map if data_type in [DataType.IA, DataType.FIXATIONS] else {}

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering."""
        return []

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """IITBHGC-specific processing steps."""

        # Process IA and FIXATIONS data
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = data_dict[data_type]

            # Add unique trial ID
            df[Fields.UNIQUE_TRIAL_ID] = (
                df[Fields.SUBJECT_ID].astype(str)
                + '_'
                + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
            )

            # Parse list columns
            list_columns = [
                'CURRENT_FIX_X',
                'CURRENT_FIX_DURATION',
                'CURRENT_FIX_INDEX',
                'IA_LABEL',
            ]
            df[list_columns] = df[list_columns].map(literal_eval)
            df = df.explode(list_columns).reset_index(drop=True)

            # Clean text
            df['paragraph'] = (
                df['paragraph']
                .str.replace('↵', ' ', regex=False)
                .str.split()
                .str.join(' ')
            )
            df['label'] = (df['true_labels'] == df['annotator_labels']).astype(int)
            df['CURRENT_FIX_X'] = df['CURRENT_FIX_X'].apply(
                lambda x: int(x) - 1 if x != '.' else -1
            )

            if data_type == DataType.FIXATIONS:
                df['CURRENT_FIX_Y'] = 0
                df['CURRENT_FIX_INDEX'] = df['CURRENT_FIX_INDEX'].astype(int)

            data_dict[data_type] = df

        # Compute reading measures
        rm_df = self.compute_word_level_reading_measures(
            data_dict['fixations'], data_dict['ia']
        )
        data_dict['ia'] = rm_df

        # Add IA features to fixation data
        logger.info('Adding IA report features to fixation data...')
        data_dict[DataType.FIXATIONS], data_dict[DataType.IA] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict[DataType.IA],
                data_dict[DataType.FIXATIONS],
            )
        )

        # Add missing features
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type == DataType.IA:
                data_dict['ia']['NEXT_FIX_INTEREST_AREA_INDEX'] = data_dict['ia'][
                    'word_index'
                ].shift(-1)
                data_dict['ia']['PREVIOUS_FIX_INTEREST_AREA_INDEX'] = data_dict['ia'][
                    'word_index'
                ].shift(1)
            else:
                data_dict['fixations']['NEXT_FIX_INTEREST_AREA_INDEX'] = 0
                data_dict['fixations']['NEXT_SAC_PEAK_VELOCITY'] = 0

            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )

        # Compute trial-level features
        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        return data_dict

    def add_ia_report_features_to_fixation_data(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge per-IA features into fixation-level data."""

        # Remove duplicates from groupby columns
        self.data_args.groupby_columns = list(
            dict.fromkeys(self.data_args.groupby_columns)
        )

        # Rename IA ID column
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # Add computed columns
        ia_df['unique_trial_id'] = (
            ia_df['participant_id'].astype(str)
            + '_'
            + ia_df['unique_paragraph_id'].astype(str)
        )
        ia_df['word_length'] = ia_df['IA_LABEL'].str.len()
        ia_df['TRIAL_IA_COUNT'] = ia_df.groupby('unique_trial_id')[
            'unique_trial_id'
        ].transform('count')

        surp_extractor = get_surp_extractor(
            extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name='gpt2'
        )
        nlp = spacy.load('en_core_web_sm')

        # Process metrics
        def process_group(group):
            sentence = group.iloc[0].paragraph
            metrics = get_metrics(
                target_text=sentence,
                surp_extractor=surp_extractor,
                parsing_model=nlp,
                parsing_mode='re-tokenize',
                add_parsing_features=True,
                language='en',
            )
            metrics['unique_paragraph_id'] = group['unique_paragraph_id'].iloc[0]
            metrics[Fields.FIXATION_REPORT_IA_ID_COL_NAME] = metrics['Token_idx']
            return metrics

        metrics_list = [
            process_group(group)
            for _, group in tqdm(
                ia_df.groupby(Fields.UNIQUE_PARAGRAPH_ID), desc='Processing metrics'
            )
        ]
        metrics_df = pd.concat(metrics_list, ignore_index=True)

        # Merge metrics
        ia_df[Fields.UNIQUE_TRIAL_ID] = (
            ia_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + ia_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )
        merge_keys = {'unique_paragraph_id', Fields.FIXATION_REPORT_IA_ID_COL_NAME}
        drop_keys = (set(metrics_df.columns) & set(ia_df.columns)) - merge_keys

        ia_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = ia_df['word_index']
        ia_df = ia_df.merge(
            metrics_df.drop(columns=list(drop_keys)), on=list(merge_keys), how='left'
        )

        # Rename columns
        column_renames = {
            'POS': 'universal_pos',
            'Length': 'word_length_no_punctuation',
            'Wordfreq_Frequency': 'wordfreq_frequency',
            'subtlex_Frequency': 'subtlex_frequency',
            'Reduced_POS': 'ptb_pos',
            'Head_word_idx': 'head_word_index',
            'Dependency_Relation': 'dependency_relation',
            'Entity': 'entity_type',
            'gpt2_Surprisal': 'gpt2_surprisal',
            'gpt2': 'gpt2_surprisal',
            'Head_Direction': 'head_direction',
            'Is_Content_Word': 'is_content_word',
            'n_Lefts': 'left_dependents_count',
            'n_Rights': 'right_dependents_count',
            'Distance2Head': 'distance_to_head',
        }
        ia_df = ia_df.rename(columns=column_renames)

        # Add default columns efficiently
        zero_columns = [
            'start_of_line',
            'end_of_line',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_SELECTIVE_REGRESSION_PATH_DURATION',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_LEFT',
            'IA_RIGHT',
            'IA_TOP',
            'IA_BOTTOM',
            'IA_REGRESSION_PATH_DURATION',
            'IA_REGRESSION_OUT_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'IA_FIRST_FIX_PROGRESSIVE',
            'normalized_ID',
            'IA_FIRST_RUN_FIXATION_COUNT',
            'IA_LAST_RUN_FIXATION_COUNT',
        ]
        ia_df[zero_columns] = 0

        # Computed columns
        ia_df['IA_FIRST_RUN_DWELL_TIME'] = ia_df['FPRT']
        ia_df['IA_FIRST_RUN_FIXATION_DURATION'] = ia_df['FPRT']
        ia_df['IA_DWELL_TIME'] = ia_df['FD']
        ia_df['IA_DWELL_TIME_%'] = ia_df.groupby('unique_trial_id')[
            'IA_DWELL_TIME'
        ].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        ia_df['PARAGRAPH_RT'] = ia_df.groupby(Fields.UNIQUE_PARAGRAPH_ID)[
            'IA_DWELL_TIME'
        ].transform('sum')
        ia_df['IA_SKIP'] = (ia_df['Fix'] > 0).astype(int)
        ia_df['total_skip'] = (ia_df['Fix'] > 0).astype(int)
        ia_df['IA_FIXATION_COUNT'] = ia_df['TFC']
        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'IA_FIXATION_COUNT'
        ].transform(lambda x: x / np.sum(x))
        ia_df['IA_FIRST_FIXATION_DURATION'] = ia_df['FFD']
        ia_df['IA_SINGLE_FIXATION_DURATION'] = ia_df['SFD']
        ia_df['IA_RUN_COUNT'] = ia_df['TFC']
        ia_df['IA_REGRESSION_OUT_FULL_COUNT'] = ia_df['TRC_out']

        # Fixation defaults
        fix_df['CURRENT_FIX_PUPIL'] = 0
        fix_df['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] = (
            fix_df['IA_LABEL'] == '.'
        ).astype(int)
        fix_df[['NEXT_SAC_DURATION', 'NEXT_SAC_AVG_VELOCITY', 'NEXT_SAC_AMPLITUDE']] = 0
        fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = fix_df['CURRENT_FIX_X'].fillna(-1)

        # IA defaults for spatial/angular features
        spatial_columns = [
            'NEXT_SAC_START_X',
            'NEXT_SAC_END_X',
            'NEXT_SAC_END_Y',
            'NEXT_SAC_START_Y',
            'PREVIOUS_FIX_DISTANCE',
            'NEXT_SAC_ANGLE',
            'NEXT_FIX_ANGLE',
            'NEXT_FIX_DISTANCE',
            'PREVIOUS_FIX_ANGLE',
        ]
        ia_df[spatial_columns] = 0

        # Merge fixations with IA features
        merge_keys = set(
            self.data_args.groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
        )
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - merge_keys
        _ia_df = ia_df.drop(columns=list(dup_cols))

        if 'normalized_part_ID' in fix_df.columns:
            fix_df = fix_df.drop(columns='normalized_part_ID')

        enriched_fix_df = fix_df.merge(
            _ia_df,
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        # Add word count
        num_of_words_in_trials = ia_df.groupby(self.data_args.groupby_columns).size()
        num_of_words_in_trials.name = 'num_of_words_in_trial'
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials,
            on=self.data_args.groupby_columns,
            how='left',
        )

        return enriched_fix_df, ia_df
