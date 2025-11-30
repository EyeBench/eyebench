from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pyreadr
import spacy
from loguru import logger
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from tqdm import tqdm

from src.configs.constants import DataType, Fields
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.data.utils import (
    add_missing_features,
    compute_trial_level_features,
    replace_missing_values,
)

tqdm.pandas()
logger.add('logs/preprocessing.log', level='INFO')


class MECOProcessor(DatasetProcessor):
    """Processor MECO dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nlp = None
        self._surp_extractor = None

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """MECO-specific processing steps"""

        # add unique trial IDs and merge labels
        labels = self._load_labels().rename(columns={'uniform_id': Fields.SUBJECT_ID})

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            df = data_dict[data_type]

            df[Fields.UNIQUE_TRIAL_ID] = (
                df[Fields.SUBJECT_ID].astype(str)
                + '_'
                + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
            )

            df = df.merge(labels, on='participant_id', validate='many_to_one')
            df = df[~df['lextale'].isna()]

            data_dict[data_type] = df

        # add IA features to fixations
        data_dict['fixations'], data_dict['ia'] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict['ia'],
                data_dict['fixations'],
            )
        )

        # add missing features
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )
            data_dict[data_type] = data_dict[data_type].assign(
                normalized_ID=(
                    data_dict[data_type]['IA_ID'] - data_dict[data_type]['IA_ID'].min()
                )
                / (
                    data_dict[data_type]['IA_ID'].max()
                    - data_dict[data_type]['IA_ID'].min()
                ),
            )

        # compute trial-level features
        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        # replace missing values
        data_dict = replace_missing_values(data_dict)

        return data_dict

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load('en_core_web_sm')
        return self._nlp

    @property
    def surp_extractor(self):
        if self._surp_extractor is None:
            self._surp_extractor = get_surp_extractor(
                extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name='gpt2'
            )
        return self._surp_extractor

    def _prepare_ia_dataframe(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # make sure order preserved
        ia_df = ia_df.sort_values(
            ['unique_trial_id', 'CURRENT_FIX_INTEREST_AREA_INDEX']
        )

        # group operations in batch
        grouped = ia_df.groupby('unique_trial_id')
        ia_df['TRIAL_IA_COUNT'] = grouped['unique_trial_id'].transform('count')
        ia_df['IA_DWELL_TIME_%'] = grouped['IA_DWELL_TIME'].transform(
            lambda x: x / np.sum(x)
        )
        ia_df['PARAGRAPH_RT'] = ia_df.groupby(Fields.UNIQUE_PARAGRAPH_ID)[
            'IA_DWELL_TIME'
        ].transform('sum')

        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'IA_FIXATION_COUNT'
        ].transform(lambda x: x / x.sum())

        # feature calculations
        ia_df['IA_FIRST_FIX_PROGRESSIVE'] = (ia_df['firstfix.sac.in'] > 0).astype(int)
        ia_df['IA_RUN_COUNT'] = ia_df['nrun']
        ia_df['IA_SELECTIVE_REGRESSION_PATH_DURATION'] = ia_df['firstrun.gopast.sel']
        ia_df['IA_SKIP'] = ia_df['total_skip']
        ia_df['IA_FIRST_FIX_PROGRESSIVE'] = (ia_df['firstfix.sac.in'] > 0).astype(int)
        ia_df['word_length'] = ia_df['IA_LABEL'].str.len()

        # not really the feature but better than 0, it is a binary indicator only
        ia_df['IA_REGRESSION_IN_COUNT'] = ia_df['IA_REGRESSION_IN']
        ia_df['IA_REGRESSION_OUT_FULL_COUNT'] = ia_df['IA_REGRESSION_OUT']
        ia_df['IA_REGRESSION_OUT_COUNT'] = ia_df['IA_REGRESSION_OUT']
        ia_df['IA_ID'] = ia_df[Fields.FIXATION_REPORT_IA_ID_COL_NAME]

        # missing columns
        zero_cols = [
            'NEXT_FIX_INTEREST_AREA_INDEX',
            'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
            'start_of_line',
            'end_of_line',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_REGRESSION_PATH_DURATION',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_LEFT',
            'IA_RIGHT',
            'IA_TOP',
            'IA_BOTTOM',
            'NEXT_SAC_DURATION',
            'NEXT_SAC_AVG_VELOCITY',
            'NEXT_SAC_AMPLITUDE',
            'NEXT_SAC_END_X',
            'NEXT_SAC_START_X',
            'NEXT_SAC_START_Y',
            'NEXT_SAC_END_Y',
        ]
        ia_df[zero_cols] = 0

        return ia_df

    def _process_metrics_batch(self, groups: list[pd.DataFrame]) -> pd.DataFrame:
        metrics_list = []

        for group in tqdm(groups, desc='Sequential metric extraction'):
            try:
                sentence = group['paragraph'].iloc[0]
                metrics = get_metrics(
                    target_text=sentence,
                    surp_extractor=self.surp_extractor,
                    parsing_model=self.nlp,
                    parsing_mode='re-tokenize',
                    add_parsing_features=True,
                    language='en',
                )
                metrics['unique_paragraph_id'] = group['unique_paragraph_id'].iloc[0]
                metrics['participant_id'] = group['participant_id'].iloc[0]
                metrics[Fields.FIXATION_REPORT_IA_ID_COL_NAME] = (
                    metrics['Token_idx'] + 1
                )
                metrics_list.append(metrics)
            except Exception as e:
                logger.warning(
                    f'Error processing group {group["unique_paragraph_id"].iloc[0]}: {e}'
                )
                continue

        return (
            pd.concat(metrics_list, ignore_index=True)
            if metrics_list
            else pd.DataFrame()
        )

    def _merge_metrics_to_ia(
        self, ia_df: pd.DataFrame, metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        merge_keys = ['unique_trial_id', Fields.FIXATION_REPORT_IA_ID_COL_NAME]

        metrics_df[Fields.UNIQUE_TRIAL_ID] = (
            metrics_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + metrics_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )

        drop_keys = (set(metrics_df.columns) & set(ia_df.columns)) - set(merge_keys)
        cols_to_drop = list(drop_keys) + ['Morph']
        cols_to_drop = [c for c in cols_to_drop if c in metrics_df.columns]

        metrics_clean = metrics_df.drop(columns=cols_to_drop).drop_duplicates()

        ia_df = ia_df.merge(
            metrics_clean,
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )

        rename_map = {
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
        ia_df = ia_df.rename(columns=rename_map)

        return ia_df

    def add_ia_report_features_to_fixation_data(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Deduplicate groupby columns
        self.data_args.groupby_columns = list(
            dict.fromkeys(self.data_args.groupby_columns)
        )

        ia_df = self._prepare_ia_dataframe(ia_df)

        # fix misalignments in meco
        # wrong paragraph ids for participant sp_36 from 5 to 12 (off by one)
        key_value_map = {
            'sp_36': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6},
            'gr_45': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8},
            'it_25': {10: 11, 9: 10, 7: 8, 6: 7},
            'se_38': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6, 4: 5},
        }
        for participant_id, mapping in key_value_map.items():
            mask = ia_df.participant_id == participant_id
            fix_mask = fix_df.participant_id == participant_id
            for old_id, new_id in sorted(mapping.items(), reverse=True):
                ia_df.loc[
                    mask & (ia_df.unique_paragraph_id == old_id), 'unique_paragraph_id'
                ] = new_id
                fix_df.loc[
                    fix_mask & (fix_df.unique_paragraph_id == old_id),
                    'unique_paragraph_id',
                ] = new_id

        ia_df['unique_trial_id'] = (
            ia_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + ia_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )
        fix_df['unique_trial_id'] = (
            fix_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + fix_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )

        # load stimuli because ia_df seems to contain different number of aois
        # in this way we can be sure to that all paragraphs are what MECO authors provide
        stimuli_df = pd.read_csv(
            'data/MECOL2/stimuli/stimuli.csv', engine='python', encoding='latin1'
        )
        stimuli_df = stimuli_df.rename(
            columns={'trialid': 'unique_paragraph_id', 'text': 'paragraph'},
        )
        stimuli_df = stimuli_df.drop(columns=['question1', 'question2'])
        ia_df = ia_df.merge(
            stimuli_df,
            on='unique_paragraph_id',
            validate='many_to_one',
        )
        fix_df = fix_df.merge(
            stimuli_df,
            on='unique_paragraph_id',
            validate='many_to_one',
        )

        groups = [group for _, group in ia_df.groupby('unique_paragraph_id')]
        metrics_df = self._process_metrics_batch(groups)

        ia_df = self._merge_metrics_to_ia(ia_df, metrics_df)

        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ].shift(-1)
        fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ].fillna(-1)

        if 'normalized_part_ID' in fix_df.columns:
            if fix_df['normalized_part_ID'].isna().any():
                logger.info('normalized_part_ID contains NaNs; dropping it.')
                fix_df = fix_df.drop(columns='normalized_part_ID')

        merge_keys = self.data_args.groupby_columns + [
            Fields.FIXATION_REPORT_IA_ID_COL_NAME
        ]
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - set(merge_keys)
        _ia_df = ia_df.drop(columns=list(dup_cols))
        enriched_fix_df = fix_df.merge(
            _ia_df.drop_duplicates(subset=merge_keys, keep='first'),
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )

        num_of_words = ia_df.groupby(self.data_args.groupby_columns).size()
        num_of_words.name = 'num_of_words_in_trial'
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words,
            on=self.data_args.groupby_columns,
            how='left',
            validate='many_to_one',
        )

        # convert types
        float_cols = ['TOWRE_word', 'TOWRE_nonword']
        for col in float_cols:
            if col in enriched_fix_df.columns:
                enriched_fix_df[col] = enriched_fix_df[col].astype(float)
            if col in ia_df.columns:
                ia_df[col] = ia_df[col].astype(float)

        enriched_fix_df['TRIAL_IA_COUNT'] = enriched_fix_df['TRIAL_IA_COUNT'].fillna(0)

        return enriched_fix_df, ia_df

    def get_column_map(self, data_type: DataType) -> dict:
        column_maps = {
            DataType.IA: {
                'uniform_id': Fields.SUBJECT_ID,
                'trialid': Fields.UNIQUE_PARAGRAPH_ID,
                'skip': 'total_skip',
                'wordnum': 'IA_ID',
                'word': 'IA_LABEL',
                'nfix': 'IA_FIXATION_COUNT',
                'reg.in': 'IA_REGRESSION_IN',
                'reg.out': 'IA_REGRESSION_OUT',
                'dur': 'IA_DWELL_TIME',
                'firstrun.nfix': 'IA_FIRST_RUN_FIXATION_COUNT',
                'firstrun.dur': 'IA_FIRST_RUN_DWELL_TIME',
                'firstfix.launch': 'IA_FIRST_RUN_LAUNCH_SITE',
                'firstfix.land': 'IA_FIRST_RUN_LANDING_POSITION',
                'firstfix.dur': 'IA_FIRST_FIXATION_DURATION',
            },
            DataType.FIXATIONS: {
                'xn': 'CURRENT_FIX_X',
                'yn': 'CURRENT_FIX_Y',
                'dur': 'CURRENT_FIX_DURATION',
                'uniform_id': Fields.SUBJECT_ID,
                'trialid': Fields.UNIQUE_PARAGRAPH_ID,
                'fixid': 'CURRENT_FIX_INDEX',
                'start': 'CURENT_FIX_START',
                'stop': 'CURRENT_FIX_END',
                'ps': 'CURRENT_FIX_PUPIL_SIZE',
                'blink': 'CURRENT_FIX_BLINK_AROUND',
                'word': 'CURRENT_FIX_INTEREST_AREA_LABEL',
                'ianum': 'CURRENT_FIX_INTEREST_AREA_INDEX',
                'ia': 'CURRENT_FIX_LABEL',
                'ia.fix': 'CURRENT_FIX_INTEREST_AREA_FIX_COUNT',
                'ia.runid': 'CURRENT_FIX_INTEREST_AREA_ID',
            },
        }

        return column_maps.get(data_type, {})

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        return []

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_labels():
        labels_w1 = pyreadr.read_r(
            'data/MECOL2W1/demographics/joint.ind.diff.l2.rda',
        )['joint_id']
        labels_w2 = pyreadr.read_r(
            'data/MECOL2W2/demographics/joint.ind.diff.l2.w2.rda',
        )['joint_id_w2']

        return pd.concat([labels_w1, labels_w2], axis=0).reset_index(drop=True)
