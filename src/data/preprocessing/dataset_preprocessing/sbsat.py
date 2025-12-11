from __future__ import annotations

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
from src.data.utils import (
    add_missing_features,
    compute_trial_level_features,
    replace_missing_values,
)

tqdm.pandas()
logger.add('logs/preprocessing.log', level='INFO')


class SBSATProcessor(DatasetProcessor):
    """Processor for SBSAT dataset"""

    N_QUESTION_DUPLICATES = 5

    # Encoding fixes
    ENCODING_MAP = {
        '\x92': "'",
        '\x93': '"',
        '\x94': '"',
        '\x97': '—',
    }

    # AOI label corrections: (paragraph_id, aoi_id, correct_label)
    AOI_FIXES = [
        # https://github.com/ahnchive/SB-SAT/blob/master/stimuli/reading%20screenshot/reading-dickens-3.png
        ('reading-dickens-3', 45, 'Sempere &'),
        # https://github.com/ahnchive/SB-SAT/blob/master/stimuli/reading%20screenshot/reading-dickens-5.png
        ('reading-dickens-5', 112, 'Mr.'),
        ('reading-dickens-5', 113, 'Dickens'),
        # https://github.com/ahnchive/SB-SAT/blob/master/stimuli/reading%20screenshot/reading-flytrap-3.png
        ('reading-flytrap-3', 30, 'Burdon-'),
        ('reading-flytrap-3', 31, "Sanderson's"),
        # https://github.com/ahnchive/SB-SAT/blob/master/stimuli/reading%20screenshot/reading-genome-2.png
        ('reading-genome-2', 70, 'species—'),
        ('reading-genome-2', 71, 'in'),
        # https://github.com/ahnchive/SB-SAT/blob/master/stimuli/reading%20screenshot/reading-genome-3.png
        ('reading-genome-3', 45, 'gee-'),
        ('reading-genome-3', 46, 'whiz,'),
    ]

    # Linguistic feature column renames
    FEATURE_RENAMES = {
        'POS': 'universal_pos',
        'Length': 'word_length_no_punctuation',
        'Wordfreq_Frequency': 'wordfreq_frequency',
        'subtlex_Frequency': 'subtlex_frequency',
        'Reduced_POS': 'ptb_pos',
        'Head_word_idx': 'head_word_index',
        'Dependency_Relation': 'dependency_relation',
        'Entity': 'entity_type',
        'gpt2_Surprisal': 'gpt2_surprisal',
        'Head_Direction': 'head_direction',
        'Is_Content_Word': 'is_content_word',
        'n_Lefts': 'left_dependents_count',
        'n_Rights': 'right_dependents_count',
        'Distance2Head': 'distance_to_head',
    }

    def add_ia_report_features_to_fixation_data(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge per-IA (interest area) features into the fixation-level data.
        Result: one row per fixation, enriched with IA-level attributes.
        """
        # Prepare IA dataframe
        ia_df = self._prepare_ia_dataframe(ia_df)

        # Prepare fixation dataframe
        fix_df = self._prepare_fixation_dataframe(fix_df, ia_df)

        # Load and merge labels
        ia_df, fix_df = self._merge_trial_labels(ia_df, fix_df)

        # Add IA features
        ia_df = self._add_ia_features(ia_df)

        # Compute linguistic features
        ia_df = self._add_linguistic_features(ia_df)

        # Finalize and merge
        fix_df, ia_df = self._finalize_dataframes(ia_df, fix_df)

        return fix_df, ia_df

    def _prepare_ia_dataframe(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare IA dataframe with metadata and reindexing"""
        ia_df = ia_df.copy()
        ia_df['old_original_word_index'] = ia_df['word_index']

        # extract story metadata
        ia_df['story_name'] = ia_df['unique_paragraph_id'].str.extract(
            r'reading-(.*)-\d+'
        )[0]
        ia_df['page_number'] = (
            ia_df['unique_paragraph_id'].str.extract(r'reading-.*-(\d+)')[0].astype(int)
        )

        # sort and reconstruct full paragraphs
        ia_df = ia_df.sort_values(
            by=[
                'participant_id',
                'story_name',
                'page_number',
                'old_original_word_index',
            ]
        )

        full_paragraphs = ia_df.groupby(['participant_id', 'story_name']).agg(
            {
                'word': lambda x: ' '.join(x),
                'old_original_word_index': lambda x: list(x),
            }
        )

        ia_df = ia_df.merge(
            full_paragraphs,
            how='left',
            on=['participant_id', 'story_name'],
            suffixes=('', '_combined'),
        ).rename(
            columns={
                'word_combined': 'full_paragraph',
                'old_original_word_index_combined': 'full_old_original_word_index',
            }
        )

        # Update paragraph ID and reindex
        ia_df['unique_paragraph_id'] = ia_df['story_name']
        ia_df = self._reindex_by_group(
            ia_df,
            ['participant_id', 'unique_paragraph_id'],
            ['page_number', 'word_index'],
            'word_index',
        )

        # Rename IA ID column
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        return ia_df

    def _prepare_fixation_dataframe(
        self, fix_df: pd.DataFrame, ia_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare fixation dataframe with metadata and reindexing"""
        fix_df = fix_df.copy()
        fix_df['old_original_word_index'] = fix_df['word_index']

        # Extract story metadata
        fix_df['story_name'] = fix_df['unique_paragraph_id'].str.extract(
            r'reading-(.*)-\d+'
        )[0]
        fix_df['page_number'] = (
            fix_df['unique_paragraph_id']
            .str.extract(r'reading-.*-(\d+)')[0]
            .astype(int)
        )
        fix_df['unique_paragraph_id'] = fix_df['story_name']

        # Reset index and reindex by group
        fix_df = fix_df.reset_index().rename(
            columns={'index': 'original_fixation_index'}
        )
        fix_df = self._reindex_by_group(
            fix_df,
            ['participant_id', 'unique_paragraph_id'],
            ['page_number', 'original_fixation_index'],
            'CURRENT_FIX_INDEX',
        )

        # Sort and merge full paragraphs
        fix_df = fix_df.sort_values(
            by=['participant_id', 'story_name', 'page_number', 'CURRENT_FIX_INDEX']
        )

        full_paragraphs = (
            ia_df.groupby(['participant_id', 'story_name'])
            .agg(
                {
                    'word': lambda x: ' '.join(x),
                    'old_original_word_index': lambda x: list(x),
                }
            )
            .reset_index()
        )

        fix_df = fix_df.merge(
            full_paragraphs,
            how='left',
            on=['participant_id', 'story_name'],
            suffixes=('', '_combined'),
        ).rename(
            columns={
                'word_combined': 'full_paragraph',
                'old_original_word_index_combined': 'full_old_original_word_index',
            }
        )

        return fix_df

    def _reindex_by_group(
        self,
        df: pd.DataFrame,
        groupby_cols: list[str],
        sort_cols: list[str],
        index_col: str,
    ) -> pd.DataFrame:
        """Reindex dataframe within groups"""

        def reindex_group(group):
            group = group.copy().sort_values(by=sort_cols)
            group[index_col] = range(len(group))
            return group

        return df.groupby(groupby_cols, group_keys=False).apply(reindex_group)

    def _add_ia_features(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Add all IA-level features with default values"""
        feature_defaults = [
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
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
            'NEXT_SAC_DURATION',
        ]

        # Add features with special calculations
        ia_df['IA_REGRESSION_PATH_DURATION'] = ia_df['RPD_inc']
        ia_df['IA_FIRST_RUN_FIXATION_COUNT'] = ia_df['FFID'].apply(
            lambda x: 1 if x > 0 else 0
        )
        ia_df['IA_FIRST_FIXATION_TIME'] = ia_df['FFD']
        ia_df['IA_FIRST_FIXATION_DURATION'] = ia_df['FFD']
        ia_df['IA_REGRESSION_IN_COUNT'] = ia_df['TRC_in']
        ia_df['IA_REGRESSION_IN'] = ia_df['TRC_in']
        ia_df['NEXT_SAC_END_X'] = ia_df['NEXT_SAC_START_X'] + ia_df['SL_out']
        ia_df['NEXT_SAC_END_Y'] = ia_df['NEXT_SAC_START_Y'] + ia_df['SL_out']
        ia_df['IA_REGRESSION_OUT_COUNT'] = ia_df['TRC_out']
        ia_df['IA_REGRESSION_OUT'] = ia_df['TRC_out']
        ia_df['IA_DWELL_TIME'] = ia_df['TFT']
        ia_df['IA_DWELL_TIME_%'] = ia_df.groupby('unique_trial_id')[
            'IA_DWELL_TIME'
        ].transform(lambda x: x / x.sum())
        ia_df['IA_FIRST_RUN_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'FFD'
        ].transform(lambda x: x / x.sum())
        ia_df['PARAGRAPH_RT'] = ia_df.groupby(Fields.UNIQUE_PARAGRAPH_ID)[
            'TFT'
        ].transform('sum')
        ia_df['IA_FIXATION_COUNT'] = ia_df['TFC']
        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')['TFC'].transform(
            lambda x: x / x.sum()
        )
        ia_df['IA_RUN_COUNT'] = ia_df['word_index']
        ia_df['IA_FIRST_FIXATION_DURATION'] = ia_df['FFD']

        # Add all default features
        for feature in feature_defaults:
            if feature not in ia_df.columns:
                ia_df[feature] = 0

        return ia_df

    def _merge_trial_labels(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge trial labels and duplicate for multiple questions"""
        # Load labels
        labels_df = pd.read_csv('data/SBSAT/labels/18sat_trialfinal.csv').rename(
            columns={
                'RECORDING_SESSION_LABEL': Fields.SUBJECT_ID,
                'page_name': Fields.UNIQUE_PARAGRAPH_ID,
            }
        )

        # Drop conflicting columns from fixation data
        fix_df = fix_df.drop(columns=['correct_answer', 'answer'], errors='ignore')

        # Duplicate both dataframes for multiple questions
        fix_df = self._duplicate_df(fix_df, self.N_QUESTION_DUPLICATES)
        fix_df['unique_trial_id'] = (
            fix_df['participant_id'].astype(str)
            + '_'
            + fix_df['unique_paragraph_id'].astype(str)
            + '_'
            + fix_df['question_index'].astype(str)
        )

        # Prepare labels
        labels_df = labels_df.loc[labels_df['correct_answer'] != -99].copy()
        labels_df['question_index'] = labels_df['unique_paragraph_id'].apply(
            lambda x: int(x.split('-')[-1])
        )
        labels_df[Fields.UNIQUE_PARAGRAPH_ID] = labels_df['unique_paragraph_id'].apply(
            lambda x: x.split('-')[1]
        )
        labels_df.drop('page', axis=1, inplace=True, errors='ignore')

        # Merge fixation labels
        merge_keys = {'book', 'participant_id', 'question_index'}
        drop_keys = (
            (set(fix_df.columns) & set(labels_df.columns))
            - merge_keys
            - {'correct_answer', 'answer'}
        )
        fix_df = fix_df.merge(
            labels_df.drop(columns=list(drop_keys)).drop_duplicates(),
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        # Prepare and merge IA labels
        ia_df = self._duplicate_df(ia_df, self.N_QUESTION_DUPLICATES)
        ia_df['unique_trial_id'] = (
            ia_df['participant_id'].astype(str)
            + '_'
            + ia_df['unique_paragraph_id'].astype(str)
            + '_'
            + ia_df['question_index'].astype(str)
        )
        ia_df['book_name'] = ia_df['story_name']
        ia_df['page'] = ia_df['page_number']

        merge_keys = {'book_name', 'participant_id', 'question_index'}
        drop_keys = (
            (set(ia_df.columns) & set(labels_df.columns))
            - merge_keys
            - {'correct_answer', 'answer'}
        )
        ia_df = ia_df.merge(
            labels_df.drop(columns=list(drop_keys)).drop_duplicates(),
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        # Merge additional labels
        _labels_df = pd.read_csv('data/SBSAT/labels/18sat_labels.csv').rename(
            columns={
                'subj': Fields.SUBJECT_ID,
                'page_name': Fields.UNIQUE_PARAGRAPH_ID,
                'book': 'book_name',
            }
        )

        merge_keys = {'book_name', 'participant_id'}
        for df in [fix_df, ia_df]:
            drop_keys = (
                (set(df.columns) & set(_labels_df.columns))
                - merge_keys
                - {'correct_answer', 'answer'}
            )
            df_merged = df.merge(
                _labels_df.drop(columns=list(drop_keys)).drop_duplicates(),
                on=list(merge_keys),
                how='left',
            )
            if df is fix_df:
                fix_df = df_merged
            else:
                ia_df = df_merged

        return ia_df, fix_df

    def _duplicate_df(self, df: pd.DataFrame, n_duplicates: int) -> pd.DataFrame:
        """Duplicate dataframe with question_index"""
        df_list = [
            df.copy().assign(question_index=i) for i in range(1, n_duplicates + 1)
        ]
        return pd.concat(df_list, ignore_index=True)

    def _add_linguistic_features(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        """Add linguistic features using spacy and language models"""

        # Initialize models
        logger.info('Initializing linguistic models...')
        surp_extractor = get_surp_extractor(
            extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name='gpt2'
        )
        nlp = spacy.load('en_core_web_sm')

        # Process groups
        logger.info('Computing linguistic features...')
        groups = list(ia_df.groupby('unique_trial_id'))
        metrics_list = [
            self._process_linguistic_group(group, surp_extractor, nlp)
            for _, group in tqdm(groups, desc='Extracting features')
        ]

        # Combine and prepare metrics
        metrics_df = pd.concat(metrics_list, ignore_index=True)
        metrics_df = self._duplicate_df(metrics_df, self.N_QUESTION_DUPLICATES)
        metrics_df['unique_trial_id'] = (
            metrics_df['unique_trial_id']
            + '_'
            + metrics_df['question_index'].astype(str)
        )

        # Merge with IA data
        ia_df['IA_ID'] = ia_df['word_index']
        merge_keys = {'unique_trial_id', 'IA_ID'}
        drop_keys = (set(ia_df.columns) & set(metrics_df.columns)) - merge_keys

        ia_df = ia_df.merge(
            metrics_df.drop(columns=list(drop_keys) + ['Morph']).drop_duplicates(),
            on=list(merge_keys),
            how='left',
            validate='one_to_one',
        )

        # Rename linguistic features
        ia_df = ia_df.rename(columns=self.FEATURE_RENAMES)

        return ia_df

    def _process_linguistic_group(
        self, group: pd.DataFrame, surp_extractor, nlp
    ) -> pd.DataFrame:
        """Process a single group for linguistic features"""

        words = list(group['word'])
        sentence = ' '.join(words)

        metrics = get_metrics(
            target_text=sentence,
            surp_extractor=surp_extractor,
            parsing_model=nlp,
            parsing_mode='re-tokenize',
            add_parsing_features=True,
            language='en',
        )

        metrics['unique_trial_id'] = (
            f'{group["participant_id"].iloc[0]}_{group["unique_paragraph_id"].iloc[0]}'
        )
        metrics['IA_ID'] = list(group['word_index'])
        metrics['CURRENT_FIX_INTEREST_AREA_INDEX'] = list(group['word_index'])

        return metrics

    def _finalize_dataframes(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Finalize both dataframes with computed features"""
        # Add derived features to IA data
        ia_df['word_length'] = ia_df['word'].apply(len)
        ia_df['total_skip'] = ia_df['FFD'].apply(lambda x: x > 0)
        ia_df['TRIAL_IA_COUNT'] = ia_df.groupby('unique_trial_id')[
            'unique_trial_id'
        ].transform('count')
        ia_df['RC'] = (ia_df['answer'] == ia_df['correct_answer']).astype(int)
        ia_df = ia_df[~ia_df['RC'].isna()]
        ia_df['binary_difficulty'] = (
            ia_df['difficulty'] > ia_df['difficulty'].median()
        ).astype(int)
        ia_df['difficulty'] = ia_df['difficulty'].astype(float)
        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'IA_FIXATION_COUNT'
        ].transform(lambda x: x / np.sum(x) if np.sum(x) > 0 else 0)
        ia_df['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] = 0
        ia_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = ia_df['IA_ID']
        ia_df['NEXT_FIX_INTEREST_AREA_INDEX'] = (
            ia_df['CURRENT_FIX_INTEREST_AREA_INDEX'].shift(-1).fillna(-1).astype(int)
        )
        ia_df['paragraph'] = ia_df.groupby('unique_trial_id')['word'].transform(
            lambda x: ' '.join(x)
        )
        ia_df['IA_SKIP'] = (ia_df['FPF'] > 0).astype(int)
        ia_df['IA_FIRST_RUN_DWELL_TIME'] = ia_df['FPRT']
        fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = (
            fix_df['word_index'].fillna(-1).astype(int)
        )
        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = (
            fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'].shift(-1).fillna(-1).astype(int)
        )

        # Map trial-level features to fixation data
        for col in ['RC', 'difficulty', 'binary_difficulty']:
            trial_values = ia_df.groupby('unique_trial_id')[col].first()
            fix_df[col] = fix_df['unique_trial_id'].map(trial_values)

        fix_df['difficulty'] = fix_df['difficulty'].astype(float)

        # Prepare for merge
        merge_keys = set(
            self.data_args.groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
        )
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - merge_keys
        _ia_df = ia_df.drop(columns=list(dup_cols))

        # Clean problematic columns
        if (
            'normalized_part_ID' in fix_df.columns
            and fix_df['normalized_part_ID'].isna().any()
        ):
            logger.info('Dropping normalized_part_ID due to NaN values')
            fix_df = fix_df.drop(columns='normalized_part_ID')

        # Merge fixations with IA features
        enriched_fix_df = fix_df.merge(
            _ia_df, on=list(merge_keys), how='left', validate='many_to_one'
        )

        # Remove duplicate groupby columns if present
        if len(set(self.data_args.groupby_columns)) != len(
            self.data_args.groupby_columns
        ):
            logger.warning('Removing duplicate groupby_columns')
            seen = set()
            self.data_args.groupby_columns = [
                x
                for x in self.data_args.groupby_columns
                if not (x in seen or seen.add(x))
            ]

        # Add word count per trial
        num_of_words_in_trials_series = _ia_df.groupby(
            self.data_args.groupby_columns
        ).apply(len)
        num_of_words_in_trials_series.name = 'num_of_words_in_trial'
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials_series,
            on=self.data_args.groupby_columns,
            how='left',
            validate='many_to_one',
        )

        return enriched_fix_df, ia_df

    def compute_word_level_reading_measures(
        self,
        fix_df: pd.DataFrame,
        stim_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute word-level reading measures from fixation data"""
        stim_df = stim_df.copy()
        stim_df['word_index'] = stim_df['word_number'] - 1

        results = []

        for text_name in tqdm(
            fix_df['unique_paragraph_id'].unique(),
            desc='Computing word-level reading measures',
        ):
            if text_name.startswith('question'):
                continue

            logger.info(f'Processing text: {text_name}')
            aoi_df = stim_df[stim_df['filename'] == f'{text_name}.png']

            # Build text strings list
            _text_strs = ['previous_PAGE', 'next_PAGE', 'GO_TO_QUESTION'] + list(
                aoi_df['word']
            )[3 : -(aoi_df.index[-1] - aoi_df[aoi_df['word'].eq('Previous')].index)[0]]
            text_strs = [s.replace("'", "'") for s in _text_strs]

            # Special case for dickens-3
            if text_name == 'reading-dickens-3':
                text_strs = [
                    s.replace('Sempere', 'Sempere &') for s in text_strs if s != '&'
                ]

            # Process each participant
            tmp_df = fix_df[fix_df['unique_paragraph_id'] == text_name]
            for participant_id in tqdm(
                tmp_df['participant_id'].unique(), desc='Participants'
            ):
                fixations_df = tmp_df[tmp_df['participant_id'] == participant_id]

                result_df = self._compute_participant_measures(
                    text_name, participant_id, fixations_df, aoi_df
                )
                results.append(result_df)

        return pd.concat(results, ignore_index=True), fix_df

    def _compute_participant_measures(
        self,
        text_name: str,
        participant_id: str,
        fixations_df: pd.DataFrame,
        aoi_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute reading measures for a single participant-text combination"""
        # Add dummy fixation at end
        fixations_df = pd.concat(
            [
                fixations_df,
                pd.DataFrame(
                    [[0] * len(fixations_df.columns)], columns=fixations_df.columns
                ),
            ],
            ignore_index=True,
        )

        # Initialize word dictionary with measures
        text_aois = list(aoi_df['word_index'])
        text_strs = list(aoi_df['word'])

        word_dict = {}
        for word_index, word in zip(text_aois, text_strs):
            word_dict[int(word_index)] = {
                'word': word,
                'word_index': word_index,
                'FFD': 0,  # First Fixation Duration
                'SFD': 0,  # Single Fixation Duration
                'FD': 0,  # Fixation Duration
                'FPRT': 0,  # First Pass Reading Time
                'FRT': 0,  # First Run Time
                'TFT': 0,  # Total Fixation Time
                'RRT': 0,  # Re-Reading Time
                'RPD_inc': 0,  # Regression Path Duration (inclusive)
                'RPD_exc': 0,  # Regression Path Duration (exclusive)
                'RBRT': 0,  # Right-Bounded Reading Time
                'Fix': 0,
                'FPF': 0,  # First Pass Fixation
                'RR': 0,  # Re-Reading
                'FPReg': 0,  # First Pass Regression
                'TRC_out': 0,  # Total Regression Count (outgoing)
                'TRC_in': 0,  # Total Regression Count (incoming)
                'SL_in': 0,  # Saccade Length (incoming)
                'SL_out': 0,  # Saccade Length (outgoing)
                'TFC': 0,  # Total Fixation Count
                'FFID': 0,  # First Fixation Index
                'IA_FIRST_FIXATION_X': 0,
                'IA_FIRST_FIXATION_Y': 0,
                'IA_FIRST_FIXATION_PREVIOUS_FIX_IA': 0,
                'IA_FIRST_FIXATION_RUN_INDEX': 0,
                'IA_FIRST_FIXATION_PREVIOUS_IAREAS': 0,
                'NEXT_SAC_START_X_tmp': 0,
                'NEXT_SAC_START_Y_tmp': 0,
                'NEXT_SAC_START_X': 0,
                'NEXT_SAC_START_Y': 0,
            }

        # Process fixations
        right_most_word = -1
        cur_fix_word_idx = -1
        next_fix_word_idx = -1
        next_fix_dur = -1

        for index, fixation in fixations_df.iterrows():
            # Skip invalid fixations
            try:
                aoi = int(fixation['word_index'])
            except (ValueError, TypeError):
                continue

            if fixation['word'] == '.':
                continue

            # Shift fixation window
            last_fix_word_idx = cur_fix_word_idx
            cur_fix_word_idx = next_fix_word_idx
            cur_fix_dur = next_fix_dur

            next_fix_word_idx = aoi - 1
            next_fix_dur = fixation['CURRENT_FIX_DURATION']

            # Skip zero-duration fixations
            if next_fix_dur == 0:
                next_fix_word_idx = cur_fix_word_idx

            # Update rightmost position
            if right_most_word < cur_fix_word_idx:
                right_most_word = cur_fix_word_idx

            # Skip uninitialized fixations
            if cur_fix_word_idx == -1 or cur_fix_word_idx not in word_dict:
                continue

            # Update word measures
            wd = word_dict[cur_fix_word_idx]
            wd['TFT'] += int(cur_fix_dur)
            wd['TFC'] += 1

            if wd['FD'] == 0:
                wd['FD'] += int(cur_fix_dur)
                wd['FFID'] = index
                wd['IA_FIRST_FIXATION_X'] = fixation['CURRENT_FIX_X']
                wd['IA_FIRST_FIXATION_Y'] = fixation['CURRENT_FIX_Y']
                wd['IA_FIRST_FIXATION_PREVIOUS_FIX_IA'] = last_fix_word_idx
                wd['IA_FIRST_FIXATION_RUN_INDEX'] = index
                wd['IA_FIRST_FIXATION_PREVIOUS_IAREAS'] = last_fix_word_idx
            wd['NEXT_SAC_START_X_tmp'] += fixation['CURRENT_FIX_X']
            wd['NEXT_SAC_START_Y_tmp'] += fixation['CURRENT_FIX_Y']
            wd['NEXT_SAC_START_X'] = np.mean(wd['NEXT_SAC_START_X_tmp'])
            wd['NEXT_SAC_START_Y'] = np.mean(wd['NEXT_SAC_START_Y_tmp'])
            # First pass measures
            if right_most_word == cur_fix_word_idx:
                if wd['TRC_out'] == 0:
                    wd['FPRT'] += int(cur_fix_dur)
                    if last_fix_word_idx < cur_fix_word_idx:
                        wd['FFD'] += int(cur_fix_dur)
            else:
                if right_most_word < cur_fix_word_idx:
                    logger.warning('Rightmost word inconsistency detected')
                if right_most_word in word_dict:
                    word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

            # Regression tracking
            if cur_fix_word_idx < last_fix_word_idx:
                wd['TRC_in'] += 1
            if cur_fix_word_idx > next_fix_word_idx:
                wd['TRC_out'] += 1

            # Right-bounded reading time
            if cur_fix_word_idx == right_most_word:
                wd['RBRT'] += int(cur_fix_dur)

            # First run time
            if wd['FRT'] == 0 and (
                next_fix_word_idx != cur_fix_word_idx or next_fix_dur == 0
            ):
                wd['FRT'] = wd['TFT']

            # Saccade lengths
            if wd['SL_in'] == 0:
                wd['SL_in'] = cur_fix_word_idx - last_fix_word_idx
            if wd['SL_out'] == 0:
                wd['SL_out'] = next_fix_word_idx - cur_fix_word_idx

        # Finalize measures
        word_measures = []
        for word_idx, wd in sorted(word_dict.items()):
            if wd['FFD'] == wd['FPRT']:
                wd['SFD'] = wd['FFD']
            wd['RRT'] = wd['TFT'] - wd['FPRT']
            wd['FPF'] = int(wd['FFD'] > 0)
            wd['RR'] = int(wd['RRT'] > 0)
            wd['FPReg'] = int(wd['RPD_exc'] > 0)
            wd['Fix'] = int(wd['TFT'] > 0)
            wd['RPD_inc'] = wd['RPD_exc'] + wd['RBRT']
            wd['participant_id'] = participant_id
            wd['unique_paragraph_id'] = text_name
            word_measures.append(wd)

        return pd.DataFrame(word_measures)

    def fix_issues_with_aois_and_stimuli(
        self, fix_df: pd.DataFrame, stim_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Fix known issues with AOI labels and apply corrections"""
        logger.info('Fixing AOI and stimulus issues...')
        fix_df = fix_df.copy()

        # Fix text encoding issues
        for bad_char, good_char in self.ENCODING_MAP.items():
            fix_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = fix_df[
                'CURRENT_FIX_INTEREST_AREA_LABEL'
            ].str.replace(bad_char, good_char, regex=False)

        # Apply AOI label corrections
        for para_id, aoi_id, correct_label in self.AOI_FIXES:
            mask = (fix_df['unique_paragraph_id'] == para_id) & (
                fix_df['CURRENT_FIX_INTEREST_AREA_ID'] == aoi_id
            )
            fix_df.loc[mask, 'CURRENT_FIX_INTEREST_AREA_LABEL'] = correct_label

        # Add derived columns
        fix_df['word'] = fix_df['CURRENT_FIX_INTEREST_AREA_LABEL']
        fix_df['word_index'] = fix_df['CURRENT_FIX_INTEREST_AREA_ID']
        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = (
            fix_df['word_index'].shift(-1).replace([np.inf, -np.inf], 0)
        )
        fix_df['PREVIOUS_FIX_INTEREST_AREA_INDEX'] = (
            fix_df['word_index'].shift(1).replace([np.inf, -np.inf], 0)
        )
        fix_df['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] = 0
        fix_df['NEXT_SAC_AMPLITUDE'] = 0
        fix_df['NEXT_SAC_AVG_VELOCITY'] = 0
        fix_df['NEXT_SAC_DURATION'] = 0
        fix_df['start_of_line'] = 0
        fix_df['end_of_line'] = 0

        return fix_df

    def get_column_map(self, data_type: DataType) -> dict[str, str]:
        """Get column mapping for SBSAT dataset"""
        if data_type == DataType.IA:
            return {
                'FFD': 'IA_FIRST_FIXATION_DURATION',
                'SFD': 'IA_SINGLE_FIXATION_DURATION',
                'TFC': 'IA_FIXATION_COUNT',
                'TRC_in': 'IA_REGRESSION_IN_COUNT',
                'TRC_out': 'IA_REGRESSION_OUT_COUNT',
                'FD': 'IA_FIRST_FIXATION_TIME',
            }
        elif data_type == DataType.FIXATIONS:
            return {
                'page_name': 'unique_paragraph_id',
                'RECORDING_SESSION_LABEL': 'participant_id',
            }
        return {}

    def get_columns_to_keep(self) -> list:
        """Get list of columns to keep after filtering"""
        return []

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """SBSAT-specific processing steps"""
        # Filter to reading trials only
        data_dict['fixations'] = data_dict['fixations'].loc[
            data_dict['fixations']['unique_paragraph_id'].str.startswith('reading')
        ]
        data_dict['ia'] = data_dict['ia'].loc[
            data_dict['ia']['filename'].str.startswith('reading')
        ]

        # Fix AOI and stimuli issues
        data_dict['fixations'] = self.fix_issues_with_aois_and_stimuli(
            data_dict['fixations'], data_dict['ia']
        )

        # Compute word-level reading measures
        data_dict['ia'], data_dict['fixations'] = (
            self.compute_word_level_reading_measures(
                data_dict['fixations'], data_dict['ia']
            )
        )

        # Add IA features to fixation data
        data_dict['fixations'], data_dict['ia'] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict['ia'], data_dict['fixations']
            )
        )

        # Add missing features for both data types
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

            # Merge questions
            questions = (
                pd.read_csv(
                    'data/SBSAT/stimuli/combined_stimulus.csv',
                    usecols=['stimulus_type', 'sequence_num', 'question'],
                )
                .rename(
                    columns={
                        'stimulus_type': 'book_name',
                        'sequence_num': 'question_index',
                    }
                )
                .drop_duplicates()
            )
            data_dict[data_type] = data_dict[data_type].merge(
                questions,
                on=['book_name', 'question_index'],
                how='left',
                validate='many_to_one',
            )

        # Compute trial-level features
        trial_level_features = compute_trial_level_features(
            raw_fixation_data=data_dict[DataType.FIXATIONS],
            raw_ia_data=data_dict[DataType.IA],
            trial_groupby_columns=self.data_args.groupby_columns,
            processed_data_path=self.data_args.processed_data_path,
        )
        data_dict[DataType.TRIAL_LEVEL] = trial_level_features

        # Replace missing values
        data_dict = replace_missing_values(data_dict)

        return data_dict
