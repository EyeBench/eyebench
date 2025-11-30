from __future__ import annotations

from ast import literal_eval

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType
from tqdm import tqdm

from src.configs.constants import DataType, Fields
from src.data.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.data.utils import add_missing_features, compute_trial_level_features

tqdm.pandas()
logger.add('logs/preprocessing.log', level='INFO')


class CopCoProcessor(DatasetProcessor):
    """Processor for CopCo dataset"""

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """CopCo-specific processing steps"""
        participants_df = self._load_participant_labels()

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = self._process_data_type(
                data_dict[data_type], data_type, participants_df
            )
            data_dict[data_type] = df

        data_dict['fixations'], data_dict['ia'] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict['ia'], data_dict['fixations']
            )
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                trial_groupby_columns=self.data_args.groupby_columns,
                mode=data_type,
            )

        data_dict[DataType.TRIAL_LEVEL] = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )

        return data_dict

    def get_column_map(self, data_type: DataType) -> dict:
        """Get column mapping for CopCo dataset"""
        if data_type == DataType.IA:
            return {
                'char_IA_ids': 'char_IA_ids',
                'landing_position': 'landing_position',
                'number_of_fixations': 'IA_FIXATION_COUNT',
                'paragraphId': 'paragraph_id',
                'paragraphid': 'paragraph_id',
                'part': 'part',
                'sentenceId': 'sentence_id',
                'speechId': 'speech_id',
                'speechid': 'speech_id',
                'trialId': 'trial_id',
                'word': 'word',
                'wordId': 'word_id',
                'word_first_fix_dur': 'IA_FIRST_FIX_DURATION',
                'word_first_pass_dur': 'IA_FIRST_FIX_DWELL_TIME',
                'word_go_past_time': 'IA_REGRESSION_OUT_TIME',
                'word_mean_fix_dur': 'IA_DWELL_TIME',
                'word_mean_sacc_dur': 'mean_sacc_dur',
                'word_peak_sacc_velocity': 'peak_sacc_velocity',
                'word_total_fix_dur': 'IA_TOTAL_FIXATION_DURATION',
            }
        elif data_type == DataType.FIXATIONS:
            return {
                'Trial_Index_': 'trial_id',
                'paragraphid': 'paragraph_id',
                'speechid': 'speech_id',
            }

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        return []

    def _load_participant_labels(self) -> pd.DataFrame:
        """Load and prepare participant labels"""
        participants_df = pd.read_csv(self.data_args.participant_stats_path)
        participants_df = participants_df.rename(
            columns={
                'subj': 'participant_id',
                'score_reading_comprehension_test': 'RCS_score',
            }
        )
        participants_df['dyslexia'] = (
            participants_df['dyslexia'].map({'yes': 1, 'no': 0}).astype(int)
        )
        return participants_df

    def _process_data_type(
        self,
        df: pd.DataFrame,
        data_type: DataType,
        participants_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process a single data type (IA or FIXATIONS)"""
        # drop instances:
        # - paragraph_id == -1 are test trials
        # - missing values in speech_id
        # - speech_id == 1327 is a practice speech
        # https://github.com/norahollenstein/copco-processing/blob/0b3bd294a3c09e186c551c085b709423181947f9/extract_features.py#L30
        df = df[
            (df.paragraph_id != -1) & (~df['speech_id'].isna() & (df.speech_id != 1327))
        ].reset_index(drop=True)

        # Add participant IDs
        if data_type == DataType.IA:
            df['participant_id'] = (
                df['source_file'].astype(str).str.zfill(2).apply(lambda x: f'P{x}')
            )
        elif data_type == DataType.FIXATIONS:
            df['participant_id'] = df['RECORDING_SESSION_LABEL'].apply(literal_eval)
            df = self._map_character_fixations_to_words(df)

        df = self._add_unique_ids(df)
        df = self._filter_and_log_nulls(df)
        df['speech_id'] = df['speech_id'].astype('Float64').astype('Int64')
        df['paragraph_id'] = df['paragraph_id'].astype('Float64').astype('Int64')
        df['trial_id'] = df['trial_id'].astype('Int64')

        df = self._merge_participant_labels(df, participants_df)

        return df

    def _add_unique_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        speech_id_str = df['speech_id'].astype(int).astype(str)
        paragraph_id_str = df['paragraph_id'].astype(int).astype(str)
        participant_id_str = df['participant_id'].astype(str)

        df['unique_paragraph_id'] = speech_id_str + '_' + paragraph_id_str
        df['unique_trial_id'] = (
            participant_id_str + '_' + speech_id_str + '_' + paragraph_id_str
        )
        return df

    def _filter_and_log_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter null values and log warnings"""
        if df['speech_id'].isna().any():
            null_speech_df = df[df['speech_id'].isna()]
            n_trials = null_speech_df['unique_trial_id'].nunique()
            n_subjects = null_speech_df['participant_id'].nunique()
            logger.warning(
                f'WARNING: {n_trials} trials of {n_subjects} subjects have null values in speech_id'
            )
            df = df.dropna(subset=['speech_id'])

        if df['paragraph_id'].isna().any():
            null_paragraph_df = df[df['paragraph_id'].isna()]
            n_trials = null_paragraph_df['unique_trial_id'].nunique()
            logger.warning(
                f'WARNING: {n_trials} trials have null values in paragraph_id'
            )
            df = df.dropna(subset=['paragraph_id'])

        df = df[df['speech_id'] != 'UNDEFINEDnull'].reset_index(drop=True)

        return df

    def _merge_participant_labels(
        self, df: pd.DataFrame, participants_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge participant labels and validate"""
        df = df.merge(
            participants_df[['participant_id', 'dyslexia', 'RCS_score']],
            on='participant_id',
            how='left',
        )

        if df['dyslexia'].isna().any():
            n_participants = df[df['dyslexia'].isna()]['participant_id'].nunique()
            logger.warning(
                f'WARNING: {n_participants} participants have null values in dyslexia'
            )

        if df['RCS_score'].isna().any():
            n_participants = df[df['RCS_score'].isna()]['participant_id'].nunique()
            logger.warning(
                f'WARNING: {n_participants} participants have null values in RCS_score'
            )
            df['RCS_score'] = df['RCS_score'].fillna(-1)

        return df

    def _map_character_fixations_to_words(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map character-level fixations to word-level data"""
        # fixations were recorded on character-level interest areas
        # we need to map them to word-level interest areas
        # using the provided word2char_IA_mapping.csv file
        # https://github.com/norahollenstein/copco-processing/blob/main/char2word_mapping.py
        w2c_df = pd.read_csv(
            'data/CopCo/labels/word2char_IA_mapping.csv',
            converters={'characters': literal_eval, 'char_IA_ids': literal_eval},
        )
        w2c_df['word'] = w2c_df['word'].str.replace('\xa0', ' __NBSP__ ', regex=False)
        w2c_df['word'] = w2c_df['word'].str.split()
        w2c_df = w2c_df.explode('word', ignore_index=True)
        w2c_df['word_id'] = w2c_df.groupby(['speechId', 'paragraphId']).cumcount()

        # recalculate character IA IDs
        w2c_df = w2c_df.groupby(['speechId', 'paragraphId'], group_keys=False).apply(
            self._add_char_IA_ids
        )

        logger.info('Mapping character fixations to words...')

        # create lookup tables
        lookup_id = self._create_lookup(w2c_df, 'wordId')
        lookup_word = self._create_lookup(w2c_df, 'word')

        df['word_id'] = df.apply(self._find_mapping, axis=1, lookup=lookup_id)
        df['word'] = df.apply(self._find_mapping, axis=1, lookup=lookup_word)

        logger.info('Finished mapping character fixations to words.')
        return df

    @staticmethod
    def _add_char_IA_ids(group: pd.DataFrame) -> pd.DataFrame:
        """Calculate character IA IDs for a group"""
        lengths = group['word'].str.len()
        end_ids = lengths.cumsum()
        start_ids = end_ids.shift(1, fill_value=0) + 1
        group['char_IA_ids'] = [
            list(range(int(start), int(end) + 1))
            for start, end in zip(start_ids, end_ids)
        ]
        return group

    @staticmethod
    def _create_lookup(w2c: pd.DataFrame, lookup_key: str) -> dict:
        """Create lookup dictionary for efficient mapping"""
        lookup = {}
        for _, row in w2c.iterrows():
            key = (row['paragraphId'], row['speechId'])
            if key not in lookup:
                lookup[key] = []
            lookup[key].append((row['char_IA_ids'], row[lookup_key]))
        return lookup

    @staticmethod
    def _find_mapping(
        row: pd.Series,
        lookup: dict[tuple[int, int], list[tuple[list[int], int | str]]],
    ) -> str | int | None:
        """Find word mapping for a fixation"""
        if pd.isna(row['CURRENT_FIX_INTEREST_AREA_LABEL']):
            return None

        key = (int(row['paragraph_id']), int(row['speech_id']))
        candidates = lookup.get(key, [])

        idx = int(row['CURRENT_FIX_INTEREST_AREA_INDEX'])
        for char_ids, word_id in candidates:
            if idx in char_ids:
                return word_id

        return None

    def add_ia_report_features_to_fixation_data(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge per-IA (interest-area) features into fixation-level data"""
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # matching column names with fix_df
        ia_df = self._prepare_ia_data(ia_df)

        ia_df = self._add_ia_placeholder_columns(ia_df)
        ia_df = self._compute_text_metrics(ia_df)
        enriched_fix_df = self._merge_ia_with_fixations(ia_df, fix_df)

        return enriched_fix_df, ia_df

    def _prepare_ia_data(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare IA data by processing words and creating derived features"""
        # Process words
        ia_df['word'] = ia_df['word'].str.replace('\xa0', ' __NBSP__ ', regex=False)
        ia_df['word'] = ia_df['word'].str.split()
        ia_df = ia_df.explode('word', ignore_index=True)

        # Create word-level features
        ia_df['word_id'] = ia_df.groupby('unique_trial_id').cumcount()
        ia_df['word_length'] = ia_df['word'].str.len()
        ia_df['total_skip'] = ia_df['IA_FIRST_FIX_DURATION'].isna()
        ia_df['TRIAL_IA_COUNT'] = ia_df.groupby('unique_trial_id')[
            'unique_trial_id'
        ].transform('count')
        ia_df['TRIAL_IA_COUNT'] = ia_df['TRIAL_IA_COUNT'].fillna(0)

        # Create paragraph text
        paragraph = ia_df.groupby('unique_trial_id')['word'].apply(' '.join)
        ia_df['paragraph'] = ia_df['unique_trial_id'].map(paragraph)

        # Add paragraph-level reading time
        ia_df['PARAGRAPH_RT'] = ia_df.groupby(Fields.UNIQUE_PARAGRAPH_ID)[
            'IA_DWELL_TIME'
        ].transform('sum')
        ia_df['IA_DWELL_TIME_%'] = ia_df.groupby('unique_trial_id')[
            'IA_DWELL_TIME'
        ].transform(lambda x: x / np.sum(x))

        return ia_df

    def _add_ia_placeholder_columns(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder columns for IA features"""
        # Define all placeholder columns with their default values
        placeholder_cols = {
            'question': '',
            'IA_SKIP': ia_df['total_skip'].astype(int),
            'IA_TOP': 0,
            'IA_LEFT': 0,
            'IA_RIGHT': 0,
            'IA_BOTTOM': 0,
            'IA_REGRESSION_PATH_DURATION': 0,
            'IA_REGRESSION_IN_COUNT': 0,
            'IA_REGRESSION_OUT_FULL_COUNT': 0,
            'IA_REGRESSION_OUT_COUNT': 0,
            'IA_LAST_RUN_DWELL_TIME': 0,
            'IA_LAST_RUN_LANDING_POSITION': 0,
            'IA_SELECTIVE_REGRESSION_PATH_DURATION': 0,
            'IA_FIRST_FIXATION_VISITED_IA_COUNT': 0,
            'IA_FIRST_RUN_FIXATION_COUNT': 0,
            'IA_FIRST_RUN_LANDING_POSITION': 0,
            'IA_FIRST_FIXATION_DURATION': ia_df['IA_FIRST_FIX_DURATION'],
            'IA_FIRST_FIX_PROGRESSIVE': 0,
            'IA_LAST_FIXATION_DURATION': 0,
            'IA_FIRST_RUN_DWELL_TIME': 0,
            'IA_LAST_RUN_FIXATION_COUNT': 0,
            'start_of_line': 0,
            'end_of_line': 0,
            'PREVIOUS_FIX_DISTANCE': 0,
            'IA_FIXATION_%': 0,
            'IA_RUN_COUNT': 0,
            'NEXT_SAC_START_X': 0,
            'NEXT_SAC_START_Y': 0,
            'NEXT_SAC_END_X': 0,
            'NEXT_SAC_END_Y': 0,
            'normalized_ID': 0,
        }

        return ia_df.assign(**placeholder_cols)

    def _compute_text_metrics(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Compute linguistic metrics using surprisal and parsing"""
        # Initialize models
        model_name = 'KennethTM/gpt2-medium-danish'
        surp_extractor = get_surp_extractor(
            extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name=model_name
        )
        nlp = spacy.load('da_core_news_sm')

        def process_group(group):
            """Process a single trial group"""
            sentence = group.paragraph.iloc[0]
            metrics = get_metrics(
                target_text=sentence,
                surp_extractor=surp_extractor,
                parsing_model=nlp,
                parsing_mode='re-tokenize',
                add_parsing_features=True,
                language='da',
            )
            metrics['unique_trial_id'] = group['unique_trial_id'].iloc[0]
            metrics['word_id'] = list(group['word_id'])
            return metrics

        # Process all groups
        metrics_list = [
            process_group(group)
            for _, group in tqdm(
                ia_df.groupby('unique_trial_id'),
                desc='Computing text metrics',
                total=ia_df['unique_trial_id'].nunique(),
            )
        ]

        # Combine and merge
        metrics_df = pd.concat(metrics_list, ignore_index=True)
        merge_keys = ['unique_trial_id', 'word_id']
        drop_keys = set(metrics_df.columns) & set(ia_df.columns) - set(merge_keys)
        ia_df = ia_df.merge(metrics_df.drop(columns=list(drop_keys)), on=merge_keys)

        # Rename columns for consistency
        column_mapping = {
            'POS': 'universal_pos',
            'Length': 'word_length_no_punctuation',
            'Wordfreq_Frequency': 'wordfreq_frequency',
            'subtlex_Frequency': 'subtlex_frequency',
            'Reduced_POS': 'ptb_pos',
            'Head_word_idx': 'head_word_index',
            'Dependency_Relation': 'dependency_relation',
            'Entity': 'entity_type',
            'Head_word_index': 'head_word_index',
            'KennethTM/gpt2-medium-danish_Surprisal': 'gpt2_surprisal',
            'Head_Direction': 'head_direction',
            'Is_Content_Word': 'is_content_word',
            'n_Lefts': 'left_dependents_count',
            'n_Rights': 'right_dependents_count',
            'Distance2Head': 'distance_to_head',
        }
        ia_df = ia_df.rename(columns=column_mapping)

        return ia_df

    def _merge_ia_with_fixations(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge IA features into fixation data"""
        # Add placeholder columns to fixation data
        fix_placeholder_cols = {
            'NEXT_SAC_ANGLE': 0,
            'NEXT_FIX_ANGLE': 0,
            'NEXT_FIX_DISTANCE': 0,
            'PREVIOUS_FIX_ANGLE': 0,
            'CURRENT_FIX_PUPIL': 0,
            'NEXT_SAC_AVG_VELOCITY': 0,
        }
        fix_df = fix_df.assign(**fix_placeholder_cols)

        # Prepare merge keys
        merge_keys = self.data_args.groupby_columns + [
            Fields.FIXATION_REPORT_IA_ID_COL_NAME
        ]
        dup_cols = set(fix_df.columns) & set(ia_df.columns) - set(merge_keys)

        # Set word_id as IA ID and prepare for merge
        ia_df[Fields.FIXATION_REPORT_IA_ID_COL_NAME] = ia_df['word_id']
        _ia_df = ia_df.drop(columns=list(dup_cols))

        # Clean up fixation data
        if 'normalized_part_ID' in fix_df.columns:
            if fix_df['normalized_part_ID'].isna().any():
                logger.info('normalized_part_ID contains NaNs; dropping it.')
            fix_df = fix_df.drop(columns='normalized_part_ID')

        # Merge
        enriched_fix_df = fix_df.merge(
            _ia_df,
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )
        enriched_fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = enriched_fix_df[
            'word_id'
        ].fillna(-1)
        enriched_fix_df['IA_ID'] = enriched_fix_df['word_id'].fillna(-1)
        enriched_fix_df['TRIAL_IA_COUNT'] = enriched_fix_df['TRIAL_IA_COUNT'].fillna(0)

        # Add word count per trial
        num_of_words_in_trials = (
            _ia_df.groupby(self.data_args.groupby_columns)
            .size()
            .rename('num_of_words_in_trial')
        )
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials,
            left_on=self.data_args.groupby_columns,
            right_index=True,
            how='left',
        )

        return enriched_fix_df
