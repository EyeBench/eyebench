"""Data arguments for the eye tracking data."""

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from omegaconf import MISSING

from src.configs.constants import DatasetLanguage, Fields, PredMode
from src.configs.utils import register_data


@dataclass
class DataArgs:
    """
    A dataclass for storing configuration parameters for handling eye tracking data.

    Attributes:
        n_folds (int): Number of folds for cross-validation.
        fold_index (int): Defines the test fold. +1 is validation, rest (out of n_folds) are train.
        subject_column (str): Column that defines the subject.
        unique_item_column (str): Column that defines an item.
        ia_query (str | None): Interest area query for filtering rows.
        fixation_query (str | None): Fixation query for filtering rows.
        split_item_columns (list[str]): Defines item for train-test split grouping.
        additional_groupby_columns (list[str]): Additional columns for grouping data.
        groupby_columns (list[str]): Columns used for grouping data. Defined in __post_init__.
        stratify (str): Whether to stratify the data based on the target variable.
        processed_data_path (Path): Path to the processed data directory.
        ia_path (Path): Path to the interest area report.
        fixations_path (Path): Path to the fixation report.
        all_folds_folder (Path): Path to the folder containing all folds.
        folds_folder_name (str): Name of the folder containing the folds.
        metadata_path (Path): Path to the metadata file.
        higher_level_split (str | None): Higher level split for the data.
        base_path (Path): Base path for the data directory.
        max_scanpath_length (int): The maximum scanpath length for the eye input.
        n_questions_per_item (int): Number of questions associated with each item.

    Methods:
        __post_init__: Initializes the groupby_columns attribute based on the values of other attributes.
    """

    task: PredMode = MISSING
    n_folds: int = 4
    n_questions_per_item: int = 0
    fold_index: int = 0
    subject_column: str = Fields.SUBJECT_ID
    unique_item_column: str = Fields.UNIQUE_PARAGRAPH_ID
    unique_trial_id_column: str = Fields.UNIQUE_TRIAL_ID
    ia_query: str | None = None
    fixation_query: str | None = None
    split_item_columns: list[str | None] = field(
        default_factory=lambda: [Fields.UNIQUE_PARAGRAPH_ID]
    )

    additional_groupby_columns: list[str] = field(default_factory=list)

    # Defined in __post_init__ below
    groupby_columns: list[str] = field(default_factory=list)

    processed_data_path: Path = Path(
        ''
    )  # Path to the data directory. Can be used specify a common path for all data files.
    ia_path: Path = Path('')  # Full path to the interest area report
    fixations_path: Path = Path('')  # Full path to the fixation report
    raw_ia_path: Path = Path('')  # Full path to the raw interest area report
    raw_fixations_path: Path = Path('')
    trial_level_path: Path = Path('')  # Full path to the trial_level report
    all_folds_folder: Path = Path('data')
    folds_folder_name: str = 'folds'
    metadata_path: Path = Path('')
    stratify: str | None = None
    higher_level_split: str | None = None
    datamodule_name: str = ''
    base_path: Path = Path('')
    target_column: str = ''
    class_names: list[str] = field(default_factory=list)
    text_source: str = ''
    text_language: str = ''
    text_domain: str = ''
    text_type: str = ''
    tasks: dict[str, str] = field(default_factory=dict)
    full_dataset_name: str = ''
    max_scanpath_length: int = -1
    max_q_len: int = 0
    max_seq_len: int = 512  # not including the question
    max_tokens_in_word = 12

    def __post_init__(self):
        self.groupby_columns = (
            [self.unique_item_column, self.subject_column, self.unique_trial_id_column]
            + self.additional_groupby_columns
            + list(self.tasks.values())
        )
        # Just so they don't get dropped in filtering in preprocess
        if self.split_item_columns[0] not in self.groupby_columns:
            self.groupby_columns += self.split_item_columns

        self.datamodule_name = self.dataset_name + 'DataModule'
        self.base_path = Path('data') / self.dataset_name
        self.processed_data_path = self.base_path / 'processed'
        self.ia_path = self.processed_data_path / 'ia.feather'
        self.fixations_path = self.processed_data_path / 'fixations.feather'
        self.trial_level_path = self.processed_data_path / 'trial_level.feather'

    @property
    def dataset_name(self) -> str:
        return self.__class__.__name__.split('_')[0]

    @property
    def is_regression(self) -> bool:
        """
        Determine if the task is regression based on class_names.
        Regression tasks have a single class name (e.g., ['score'], ['lextale']).
        Classification tasks have multiple class names (e.g., ['Incorrect', 'Correct']).
        """
        return len(self.class_names) == 1

    @property
    def is_english(self) -> bool:
        """
        Return True if the dataset's text language is English.
        Handles both DatasetLanguage enum values and string names.
        """
        if isinstance(self.text_language, DatasetLanguage):
            return self.text_language == DatasetLanguage.ENGLISH
        return str(self.text_language).strip().lower() in ('english', 'en')


@register_data
@dataclass
class CopCo(DataArgs):
    """
    CopCo data.
    """

    split_item_columns: list[str] = field(
        default_factory=lambda: [
            'speech_id',
        ]
    )

    stratify: str = 'dyslexia'
    text_source: str = 'Danish Natural Reading Corpus'
    text_language: str = DatasetLanguage.DANISH
    text_domain: str = 'News'
    text_type: str = 'paragraph'

    additional_groupby_columns: list[str] = field(default_factory=lambda: [])
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.RCS: 'RCS_score',
            PredMode.TYP: 'dyslexia',
        }
    )

    max_scanpath_length: int = 484

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_path: Path = (
            self.base_path / 'precomputed_reading_measures/combined_ia.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events/combined_fixations.csv'
        )
        self.participant_stats_path: Path = (
            self.base_path / 'labels/participant_stats.csv'
        )
        self.stimuli_and_comp_results_path: Path = (
            self.base_path / 'labels/stimuli_and_comp_results.csv'
        )


@register_data
@dataclass
class CopCo_RCS(CopCo):
    """
    CopCo General Reading Comprehension
    """

    task: PredMode = PredMode.RCS
    target_column: str = 'RCS_score'
    class_names: list[str] = field(default_factory=lambda: ['score'])
    # max_seq_len: int = 350
    max_tokens_in_word: int = 15


@register_data
@dataclass
class CopCo_TYP(CopCo):
    """
    CopCo Reading Type (Dyslexia vs. Typical)
    """

    task: PredMode = PredMode.TYP
    target_column: str = 'dyslexia'
    class_names: list[str] = field(default_factory=lambda: ['Typical', 'Dyslexia'])
    # max_seq_len: int = 256
    max_tokens_in_word: int = 15


@register_data
@dataclass
class IITBHGC(DataArgs):
    """
    IITBHGC data.
    """

    text_language: str = DatasetLanguage.ENGLISH
    stratify: str = 'label'
    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.UNIQUE_PARAGRAPH_ID,
        ]
    )
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.CV: 'label',
        }
    )

    max_scanpath_length: int = 557

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_path: Path = Path(
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )


@register_data
@dataclass
class IITBHGC_CV(IITBHGC):
    """
    IITBHGC Hallucination Detection
    """

    task: PredMode = PredMode.CV
    target_column: str = 'label'
    class_names: list[str] = field(default_factory=lambda: ['unverified', 'verified'])
    # max_seq_len: int = 256
    max_tokens_in_word: int = 12


@register_data
@dataclass
class MECOL2(DataArgs):
    """
    MECOL2 data.
    """

    text_language: str = DatasetLanguage.ENGLISH
    # can't stratify due to regression label
    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.UNIQUE_PARAGRAPH_ID,
        ]
    )
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.LEX: 'lextale',
        }
    )
    max_scanpath_length: int = 802

    def __post_init__(self) -> None:
        super().__post_init__()
        self.subject_column = 'participant_id'
        self.raw_ia_path: Path = Path(
            self.base_path / 'precomputed_reading_measures' / 'combined_ia.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )


@register_data
@dataclass
class MECOL2_LEX(MECOL2):
    """
    MECOL2 Text Reading Comprehension
    """

    task: PredMode = PredMode.LEX
    target_column: str = 'lextale'
    stratify: str = 'lextale'
    class_names: list[str] = field(default_factory=lambda: ['lextale'])
    max_tokens_in_word: int = 6


@register_data
@dataclass
class MECOL2W1(DataArgs):
    """
    MECOL2W data.
    """

    text_language: str = DatasetLanguage.ENGLISH
    # can't stratify due to regression label
    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.UNIQUE_PARAGRAPH_ID,
        ]
    )
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.LEX: 'lextale',
        }
    )
    max_scanpath_length: int = 656

    def __post_init__(self) -> None:
        super().__post_init__()
        self.subject_column = 'participant_id'
        self.raw_ia_path: Path = Path(
            self.base_path / 'precomputed_reading_measures' / 'combined_ia.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )


@register_data
@dataclass
class MECOL2W2(DataArgs):
    """
    MECOL2W data.
    """

    text_language: str = DatasetLanguage.ENGLISH
    # can't stratify due to regression label
    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.UNIQUE_PARAGRAPH_ID,
        ]
    )
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.LEX: 'lextale',
        }
    )

    max_scanpath_length: int = 802

    def __post_init__(self) -> None:
        super().__post_init__()
        self.subject_column = 'participant_id'
        self.unique_item_column = 'unique_trial_id'
        self.raw_ia_path: Path = Path(
            self.base_path / 'precomputed_reading_measures' / 'combined_ia.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )


@register_data
@dataclass
class OneStop(DataArgs):
    """
    OneStop data.
    """

    n_folds: int = 10
    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.BATCH,
            Fields.ARTICLE_ID,
        ]
    )
    ia_query: str = 'practice_trial==False & question_preview==False & repeated_reading_trial==False'
    fixation_query: str = 'practice_trial==False & question_preview==False & repeated_reading_trial==False'
    stratify: str = Fields.IS_CORRECT
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.RC: Fields.IS_CORRECT,
        }
    )
    higher_level_split: str | None = Fields.BATCH
    text_source: str = 'Guardian Articles'
    text_language: str = DatasetLanguage.ENGLISH
    text_domain: str = 'News'
    text_type: str = 'paragraph'

    # Not really used, kept for eval
    additional_groupby_columns: list[str] = field(
        default_factory=lambda: [
            Fields.LIST,
            Fields.HAS_PREVIEW,
            Fields.REREAD,
        ]
    )

    max_scanpath_length: int = 815
    n_questions_per_item: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_path: Path = (
            self.base_path / 'precomputed_reading_measures' / 'ia_Paragraph.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'fixations_Paragraph.csv'
        )
        self.trial_level_paragraphs_path = (
            self.base_path / 'additional_raw' / 'trial_level_paragraphs.csv'
        )
        self.onestopqa_path = self.base_path / 'additional_raw' / 'onestop_qa.json'


@register_data
@dataclass
class OneStop_RC(OneStop):
    """
    OneStop Is Correct
    """

    task: PredMode = PredMode.RC
    target_column: str = Fields.IS_CORRECT
    class_names: list[str] = field(default_factory=lambda: ['Incorrect', 'Correct'])

    max_q_len: int = 30
    # max_seq_len: int = 280
    max_tokens_in_word: int = 10


@register_data
@dataclass
class PoTeC(DataArgs):
    """
    PoTeC data.
    """

    text_source: str = 'German Physics & Biology Textbooks'
    text_language: str = DatasetLanguage.GERMAN
    text_domain: str = 'Science Education'
    text_type: str = 'paragraph'
    stratify: str = 'DE_RC'

    split_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.UNIQUE_PARAGRAPH_ID,
        ]
    )
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.RC: 'RC',
            PredMode.DE: 'DE',
        }
    )

    additional_groupby_columns: list[str] = field(
        default_factory=lambda: [
            'question',
            'DE_RC',
        ]
    )
    n_questions_per_item: int = 3
    max_scanpath_length: int = 1483

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_path: Path = Path(
            self.base_path / 'precomputed_reading_measures' / 'combined_ia.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events' / 'combined_fixations.csv'
        )


@register_data
@dataclass
class PoTeC_DE(PoTeC):
    """
    PoTeC Background Knowledge
    """

    task: PredMode = PredMode.DE
    target_column: str = 'DE'
    class_names: list[str] = field(default_factory=lambda: ['Low', 'High'])
    # max_seq_len: int = 512
    max_tokens_in_word: int = 12


@register_data
@dataclass
class PoTeC_RC(PoTeC):
    """
    PoTeC Text Reading Comprehension
    """

    task: PredMode = PredMode.RC
    target_column: str = 'RC'
    class_names: list[str] = field(default_factory=lambda: ['Incorrect', 'Correct'])
    max_q_len: int = 40
    # max_seq_len: int = 350
    max_tokens_in_word: int = 12


@register_data
@dataclass
class SBSAT(DataArgs):
    """
    SBSAT data.
    """

    text_source: str = 'SAT Reading Passages'
    text_language: str = DatasetLanguage.ENGLISH
    text_domain: str = 'Education'
    text_type: str = 'paragraph'
    stratify: str = 'RC'
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.RC: 'RC',
            PredMode.STD: 'difficulty',
        }
    )
    max_scanpath_length: int = 1240
    n_questions_per_item: int = 5
    max_seq_len: int = 740

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_dir: Path = Path(self.base_path / 'stimuli')
        self.raw_ia_path: Path = Path(
            self.base_path / 'stimuli/' / 'combined_stimulus.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events/18sat_fixfinal.csv'
        )


@register_data
@dataclass
class SBSAT_RC(SBSAT):
    """
    SBSAT Text Reading Comprehension
    """

    task: PredMode = PredMode.RC
    target_column: str = 'RC'
    class_names: list[str] = field(default_factory=lambda: ['Incorrect', 'Correct'])
    max_q_len: int = 55
    max_tokens_in_word: int = 12


@register_data
@dataclass
class SBSAT_STD(SBSAT):
    """
    SBSAT Subjective Difficulty
    """

    task: PredMode = PredMode.STD
    target_column: str = 'difficulty'
    class_names: list[str] = field(default_factory=lambda: ['difficulty'])
    max_tokens_in_word: int = 12


def get_data_args(class_name: str) -> DataArgs | None:
    """
    Get the data path arguments class by its name.

    Args:
        class_name (str): The name of the class.

    Returns:
        DataArgs: An instance of the requested class.

    Raises:
        ValueError: If the class name is not found.
    """
    try:
        return globals()[class_name]()
    except KeyError:
        logger.error(f"Class '{class_name}' not found in src/configs/data.py.")
        return None


# Map each data_task to its config
DATA_CONFIGS_MAPPING = {
    'CopCo_TYP': CopCo_TYP,
    'CopCo_RCS': CopCo_RCS,
    'MECOL2_LEX': MECOL2_LEX,
    'SBSAT_STD': SBSAT_STD,
    'SBSAT_RC': SBSAT_RC,
    'PoTeC_DE': PoTeC_DE,
    'PoTeC_RC': PoTeC_RC,
    'IITBHGC_CV': IITBHGC_CV,
    'OneStop_RC': OneStop_RC,
}
