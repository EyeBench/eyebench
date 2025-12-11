from __future__ import annotations

import os.path

import numpy as np
import pandas as pd
import spacy
from joblib import Parallel, delayed
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


class PoTeCProcessor(DatasetProcessor):
    """Processor for PoTeC dataset with optimized performance"""

    IA_COLUMN_MAP = {
        'word': 'IA_LABEL',
        'FFD': 'IA_FIRST_FIXATION_DURATION',
        'text_surprisal_gpt2-base': 'gpt2_surprisal',
        'RPD_in': 'IA_REGRESSION_IN_COUNT',
        'document_frequency_normalized': 'wordfreq_frequency',
        'text_id': Fields.UNIQUE_PARAGRAPH_ID,
        'PoS_tag': 'universal_pos',
        'FD': 'IA_DWELL_TIME',
        'TFC': 'IA_FIXATION_COUNT',
        'FPRT': 'IA_FIRST_RUN_DWELL_TIME',
        'word_index_in_text': Fields.IA_DATA_IA_ID_COL_NAME,
        'reader_id': 'participant_id',
    }

    FIXATION_COLUMN_MAP = {
        'fixation_index': 'CURRENT_FIX_INDEX',
        'fixation_duration': 'CURRENT_FIX_DURATION',
        'fixation_position_x': 'CURRENT_FIX_X',
        'fixation_position_y': 'CURRENT_FIX_Y',
        'aoi': 'IA_ID',
        'next_saccade_duration': 'NEXT_SAC_DURATION',
        'reader_id': Fields.SUBJECT_ID,
        'item_id': Fields.UNIQUE_PARAGRAPH_ID,
    }

    LINGUISTIC_FEATURE_RENAMES = {
        'POS': 'universal_pos',
        'Length': 'word_length_no_punctuation',
        'subtlex_Frequency': 'subtlex_frequency',
        'Reduced_POS': 'ptb_pos',
        'Head_word_idx': 'head_word_index',
        'Dependency_Relation': 'dependency_relation',
        'Entity': 'entity_type',
        'Head_Direction': 'head_direction',
        'Is_Content_Word': 'is_content_word',
        'n_Lefts': 'left_dependents_count',
        'n_Rights': 'right_dependents_count',
        'Distance2Head': 'distance_to_head',
    }

    def get_column_map(self, data_type: DataType) -> dict[str, str]:
        return (
            self.IA_COLUMN_MAP if data_type == DataType.IA else self.FIXATION_COLUMN_MAP
        )

    def get_columns_to_keep(self) -> list[str]:
        return []

    def _build_stim_dict(self, stim_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {name: data for name, data in stim_df.groupby('text_name')}

    def _process_single_page(
        self,
        page_name: str,
        page_fixes: pd.DataFrame,
        stim_dict: dict[str, pd.DataFrame],
    ) -> tuple:
        if page_name not in stim_dict:
            return None, None, None, None, None

        page_stim = stim_dict[page_name]
        page_indices = page_fixes.index
        n_fixes = len(page_indices)

        word_values = np.full(n_fixes, '.', dtype=object)
        aoi_values = np.full(n_fixes, np.nan)
        start_line = np.full(n_fixes, False)
        end_line = np.full(n_fixes, False)

        fix_coords = page_fixes.loc[
            page_indices, ['CURRENT_FIX_X', 'CURRENT_FIX_Y']
        ].values

        for i, (fix_x, fix_y) in enumerate(fix_coords):
            x_match = (page_stim['start_x'] <= fix_x) & (fix_x < page_stim['end_x'])
            y_match = (page_stim['start_y'] <= fix_y) & (fix_y < page_stim['end_y'])
            aoi_match = page_stim[x_match & y_match]

            if not aoi_match.empty:
                first_match = aoi_match.iloc[0]
                word_values[i] = first_match['character']
                aoi_values[i] = int(first_match['aoi'])

        return page_indices, word_values, aoi_values, start_line, end_line

    def map_to_aois(self, fix_df: pd.DataFrame, stim_df: pd.DataFrame) -> pd.DataFrame:
        fix_df['word'] = '.'
        fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = np.nan
        fix_df['start_of_line'] = False
        fix_df['end_of_line'] = False

        stim_dict = self._build_stim_dict(stim_df)
        page_groups = list(fix_df.groupby(Fields.UNIQUE_PARAGRAPH_ID))

        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._process_single_page)(page_name, page_fixes, stim_dict)
            for page_name, page_fixes in tqdm(
                page_groups, desc='Mapping fixations to AOIs'
            )
        )

        for page_indices, word_values, aoi_values, start_line, end_line in results:
            if page_indices is not None:
                fix_df.loc[page_indices, 'word'] = word_values
                fix_df.loc[page_indices, 'CURRENT_FIX_INTEREST_AREA_INDEX'] = aoi_values
                fix_df.loc[page_indices, 'start_of_line'] = start_line
                fix_df.loc[page_indices, 'end_of_line'] = end_line

        return fix_df

    def _extract_linguistic_features_for_group(
        self, group: pd.DataFrame, nlp, surp_extractor
    ) -> pd.DataFrame:
        """Extract linguistic features for a single group"""
        words = group['IA_LABEL'].fillna('Null').tolist()
        sentence = ' '.join(words)

        metrics = get_metrics(
            target_text=sentence,
            parsing_model=nlp,
            surp_extractor=surp_extractor,
            parsing_mode='re-tokenize',
            add_parsing_features=True,
            language='de',
        )

        metrics['unique_paragraph_id'] = group['unique_paragraph_id'].iloc[0]
        metrics['unique_trial_id'] = group['unique_trial_id'].iloc[0]
        metrics[Fields.FIXATION_REPORT_IA_ID_COL_NAME] = metrics['Token_idx']

        return metrics

    def _initialize_missing_columns(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ia_zero_cols = [
            'IA_FIRST_RUN_FIXATION_COUNT',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'TRIAL_IA_COUNT',
            'IA_SELECTIVE_REGRESSION_PATH_DURATION',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_TOP',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'start_of_line',
            'end_of_line',
            'IA_LEFT',
            'IA_LAST_FIXATION_DURATION',
            'IA_REGRESSION_PATH_DURATION',
            'IA_REGRESSION_OUT_COUNT',
            'IA_FIRST_FIX_PROGRESSIVE',
            'IA_RUN_COUNT',
            'NEXT_SAC_START_Y',
            'NEXT_SAC_START_X',
            'NEXT_SAC_END_X',
            'NEXT_SAC_END_Y',
        ]
        for col in ia_zero_cols:
            ia_df[col] = 0

        fix_zero_cols = ['NEXT_SAC_AVG_VELOCITY', 'NEXT_SAC_AMPLITUDE']

        for col in fix_zero_cols:
            fix_df[col] = 0

        ia_df['normalized_ID'] = ia_df['CURRENT_FIX_INTEREST_AREA_INDEX']
        ia_df['PARAGRAPH_RT'] = ia_df.groupby('unique_trial_id')[
            'IA_DWELL_TIME'
        ].transform('sum')
        ia_df['IA_DWELL_TIME_%'] = ia_df.groupby('unique_trial_id')[
            'IA_DWELL_TIME'
        ].transform(lambda x: x / np.sum(x))
        ia_df['total_skip'] = (ia_df['IA_DWELL_TIME'] > 0).astype(int)
        ia_df['IA_SKIP'] = (ia_df['IA_DWELL_TIME'] > 0).astype(int)
        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'IA_FIXATION_COUNT'
        ].transform(lambda x: x / np.sum(x))

        return ia_df, fix_df

    def add_ia_report_features_to_fixation_data(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # remove duplicate groupby columns
        if len(set(self.data_args.groupby_columns)) != len(
            self.data_args.groupby_columns
        ):
            logger.warning('Removing duplicate groupby_columns')
            self.data_args.groupby_columns = list(
                dict.fromkeys(self.data_args.groupby_columns)
            )

        # unify IA ID column name
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # prepare IA dataframe
        ia_df['IA_LABEL'] = ia_df['IA_LABEL'].fillna('Null')
        paragraphs = ia_df.groupby('unique_trial_id')['IA_LABEL'].apply(
            lambda x: ' '.join(x)
        )
        ia_df['paragraph'] = ia_df['unique_trial_id'].map(paragraphs)
        ia_df['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] = 0

        # extract linguistic features
        logger.info(
            'Processing linguistic features in batches. This might take a while.'
        )
        nlp = spacy.load('de_core_news_sm')
        surp_extractor = get_surp_extractor(
            extractor_type=SurpExtractorType.CAT_CTX_LEFT,
            model_name='gpt2',
        )

        groups = [group for _, group in ia_df.groupby('unique_trial_id')]
        metrics_list = []
        batch_size = 100

        for i in tqdm(range(0, len(groups), batch_size), desc='Processing batches'):
            batch = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._extract_linguistic_features_for_group)(
                    g, nlp, surp_extractor
                )
                for g in groups[i : i + batch_size]
            )
            metrics_list.extend(batch)

        metrics_df = pd.concat(metrics_list, ignore_index=True)

        # merge linguistic features
        merge_keys = ['unique_trial_id', Fields.FIXATION_REPORT_IA_ID_COL_NAME]
        drop_cols = (set(ia_df.columns) & set(metrics_df.columns)) - set(merge_keys)
        ia_df = ia_df.merge(
            metrics_df.drop(columns=list(drop_cols) + ['Morph']).drop_duplicates(),
            on=merge_keys,
            how='left',
        )

        # rename linguistic feature columns
        ia_df = ia_df.rename(columns=self.LINGUISTIC_FEATURE_RENAMES)

        # initialize missing columns
        ia_df, fix_df = self._initialize_missing_columns(ia_df, fix_df)

        # add stratify column DE_RC
        ia_df['DE_RC'] = ia_df['DE'].astype(str) + '_' + ia_df['RC'].astype(str)
        fix_df['DE_RC'] = fix_df['DE'].astype(str) + '_' + fix_df['RC'].astype(str)

        # map fixations to aois
        word_aoi_df = self._read_word_aoi_df()
        fix_df = self.map_to_aois(fix_df, word_aoi_df)
        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ].shift(-1)

        # prepare merge
        merge_keys = self.data_args.groupby_columns + [
            Fields.FIXATION_REPORT_IA_ID_COL_NAME
        ]
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - set(merge_keys)
        _ia_df = ia_df.drop(columns=list(dup_cols))

        if 'normalized_part_ID' in fix_df.columns:
            fix_df = fix_df.drop(columns='normalized_part_ID')

        enriched_fix_df = fix_df.merge(
            _ia_df.drop_duplicates(subset=merge_keys, keep='first'),
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )

        num_words = (
            ia_df.groupby(self.data_args.groupby_columns)
            .size()
            .rename('num_of_words_in_trial')
        )
        enriched_fix_df = enriched_fix_df.merge(
            num_words,
            on=self.data_args.groupby_columns,
            how='left',
        )

        enriched_fix_df['normalized_ID'] = enriched_fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ]

        return enriched_fix_df, ia_df

    def _read_word_aoi_df(self) -> pd.DataFrame:
        stim_path = 'data/PoTeC/stimuli'
        dfs = []

        for file_name in os.listdir(stim_path):
            if file_name.startswith('word_aoi_'):
                file_path = os.path.join(stim_path, file_name)
                tmp_df = pd.read_csv(file_path, delimiter='\t')
                tmp_df['text_name'] = file_name.removesuffix('.tsv').split('_')[-1]
                dfs.append(tmp_df)

        return pd.concat(dfs, ignore_index=True)

    def _load_and_merge_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # participant data
        label_df = pd.read_csv(
            'data/PoTeC/labels/participant_data.tsv',
            delimiter='\t',
        ).rename(columns={'reader_id': Fields.SUBJECT_ID})

        merge_keys = [Fields.SUBJECT_ID]
        drop_cols = (set(df.columns) & set(label_df.columns)) - set(merge_keys)
        df = df.merge(
            label_df.drop(columns=list(drop_cols)),
            on=merge_keys,
            validate='many_to_one',
        )

        # response accuracy data
        text_spec_df = pd.read_csv(
            'data/PoTeC/labels/participant_response_accuracy.tsv',
            delimiter='\t',
        ).rename(
            columns={
                'reader_id': Fields.SUBJECT_ID,
                'text_id': Fields.UNIQUE_PARAGRAPH_ID,
            }
        )

        # drop existing accuracy columns if present
        acc_cols = ['mean_acc_bq', 'acc_tq_1', 'acc_tq_2', 'acc_tq_3']
        df = df.drop(columns=[col for col in acc_cols if col in df.columns])

        merge_keys = [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID]
        drop_cols = (set(df.columns) & set(text_spec_df.columns)) - set(merge_keys)
        df = df.merge(
            text_spec_df.drop(columns=list(drop_cols)),
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )

        # this is equivalent to everything correct
        df['DE'] = (df['mean_acc_bq'] > 0.9).astype(int)

        return df

    def _expand_for_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dataframe to have separate rows for each question"""
        dfs = []
        question_indices = ['tq_1', 'tq_2', 'tq_3']

        for q in tqdm(question_indices, desc='Expanding for questions'):
            df_temp = df.copy()
            df_temp['RC'] = df_temp[f'acc_{q}']
            df_temp['question_index'] = q
            df_temp['question'] = df_temp[q]
            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)
        df['unique_trial_id'] = (
            df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
            + '_'
            + df['question_index']
        )

        return df

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """PoTeC-specific processing pipeline"""
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            logger.info(f'Processing {data_type} data')
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = data_dict[data_type]

            # exclude trials without comprehension scores
            question_indices = ['tq_1', 'tq_2', 'tq_3']
            acc_columns = [f'acc_{q}' for q in question_indices]
            df = df.dropna(subset=acc_columns).copy()

            # load and merge labels
            df = self._load_and_merge_labels(df)

            # merge stimuli data
            stim_df = pd.read_csv(
                'data/PoTeC/stimuli/stimuli.tsv', delimiter='\t'
            ).rename(
                columns={
                    'reader_id': Fields.SUBJECT_ID,
                    'text_id': Fields.UNIQUE_PARAGRAPH_ID,
                }
            )
            df = df.merge(
                stim_df, on=[Fields.UNIQUE_PARAGRAPH_ID], validate='many_to_one'
            )

            # duplicate df for questions
            df = self._expand_for_questions(df)

            data_dict[data_type] = df

        data_dict['fixations'], data_dict['ia'] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict['ia'],
                data_dict['fixations'],
            )
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )

        for data_type in data_dict.keys():
            df = data_dict[data_type]
            data_dict[data_type] = df.loc[:, ~df.columns.duplicated()]

        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        data_dict = replace_missing_values(data_dict)

        return data_dict
