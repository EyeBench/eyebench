"""Constants used throughout the project."""

from enum import Enum, StrEnum
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

STATS_FOLDER = Path('data/stats')
FEATURES_CACHE_FOLDER = Path('data/cache/features')
SCANPATH_PADDING_VAL = -10

#### For numerical features used for sklearn-models (baselines)
numerical_feature_aggregations = [
    'mean',
    'std',
    'median',
    'skew',
    'kurtosis',
    'max',
    'min',
]
numerical_ia_trial_columns = [
    'landing_position',
    'IA_FIRST_FIX_DURATION',
    'IA_FIRST_FIX_DWELL_TIME',
    'IA_REGRESSION_OUT_TIME',
    'IA_DWELL_TIME',
    'IA_TOTAL_FIXATION_DURATION',
    'IA_FIXATION_COUNT',
    'mean_sacc_dur',
    'peak_sacc_velocity',
    'right_dependents_count',
    'IA_LAST_RUN_LANDING_POSITION',
    'IA_FIRST_RUN_LANDING_POSITION',
    'IA_FIRST_FIXATION_DURATION',
    'IA_RUN_COUNT',
    'IA_TOP',
    'IA_REGRESSION_IN_COUNT',
    'IA_LAST_RUN_DWELL_TIME',
    'wordfreq_frequency',
    'IA_LAST_FIXATION_DURATION',
    'IA_REGRESSION_OUT_COUNT',
    'IA_FIRST_FIXATION_VISITED_IA_COUNT',
    'IA_SELECTIVE_REGRESSION_PATH_DURATION',
    'IA_SKIP',
    'PARAGRAPH_RT',
    'left_dependents_count',
    'repeated_reading_trial',
    'word_length',
    'is_content_word',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'gpt2_surprisal',
    'IA_LAST_RUN_FIXATION_COUNT',
    'IA_FIRST_FIX_PROGRESSIVE',
    'IA_REGRESSION_PATH_DURATION',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_LEFT',
]
numerical_fixation_trial_columns = [
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_INTEREST_AREA_DWELL_TIME',
    'CURRENT_FIX_INTEREST_AREA_FIX_COUNT',
    'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
    'CURRENT_FIX_RUN_SIZE',
    'NEXT_SAC_DURATION',
    'NEXT_SAC_PEAK_VELOCITY',
    'NEXT_SAC_AMPLITUDE',
    'TRIAL_FIXATION_TOTAL',
    'CURRENT_FIX_X',
    'CURRENT_FIX_Y',
    'NEXT_FIX_DISTANCE',
    'NEXT_FIX_ANGLE',
    'right_dependents_count',
    'IA_FIRST_FIXATION_DURATION',
    'IA_RUN_COUNT',
    'IA_FIXATION_COUNT',
    'NEXT_SAC_END_Y',
    'IA_REGRESSION_IN_COUNT',
    'IA_LAST_RUN_DWELL_TIME',
    'wordfreq_frequency',
    'IA_LAST_FIXATION_DURATION',
    'normalized_incoming_regression_count',
    'IA_REGRESSION_OUT_COUNT',
    'PREVIOUS_FIX_ANGLE',
    'normalized_outgoing_regression_count',
    'IA_FIRST_FIXATION_VISITED_IA_COUNT',
    'CURRENT_FIX_PUPIL',
    'IA_SELECTIVE_REGRESSION_PATH_DURATION',
    'IA_SKIP',
    'NEXT_SAC_START_Y',
    'NEXT_SAC_ANGLE',
    'word_length',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'gpt2_surprisal',
    'IA_LAST_RUN_FIXATION_COUNT',
    'IA_DWELL_TIME',
    'IA_FIRST_FIX_PROGRESSIVE',
    'NEXT_SAC_START_X',
    'NEXT_SAC_AVG_VELOCITY',
    'IA_REGRESSION_IN_COUNT_sum',
    'IA_REGRESSION_PATH_DURATION',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_FIRST_RUN_DWELL_TIME',
    'NEXT_SAC_END_X',
    'PREVIOUS_FIX_DISTANCE',
]


gsf_features = [
    'gpt2_surprisal',
    'word_length',
    'left_dependents_count',
    'right_dependents_count',
    'distance_to_head',
    'IA_FIRST_FIXATION_DURATION',
    'IA_DWELL_TIME',
    'normalized_incoming_regression_count',
    'CURRENT_FIX_X',
    'CURRENT_FIX_Y',
    'normalized_outgoing_regression_count',
    'normalized_outgoing_progressive_count',
    'LengthCategory_normalized_IA_DWELL_TIME',
    'universal_pos_normalized_IA_DWELL_TIME',
    'LengthCategory_normalized_IA_FIRST_FIXATION_DURATION',
    'universal_pos_normalized_IA_FIRST_FIXATION_DURATION',
]


#### Dataset Language Constants
class DatasetLanguage(StrEnum):
    """
    Enum for dataset languages.

    Attributes:
        ENGLISH (str): Represents English language datasets.
        GERMAN (str): Represents German language datasets.
        DANISH (str): Represents Danish language datasets.
    """

    ENGLISH = 'English'
    GERMAN = 'German'
    DANISH = 'Danish'


class NormalizationModes(StrEnum):
    """
    Enum for normalization modes.

    Attributes:
        ALL (str): Represents the mode where data is normalized based on all trials.
        TRIAL (str): Represents the mode where data is normalized based on a trial level.
        NONE (str): Represents the mode where no data is normalized.
    """

    ALL = 'all'
    TRIAL = 'trial'
    NONE = 'none'


class RunModes(StrEnum):
    """
    Enum for run modes.

    Attributes:
        DEBUG (str): Represents the debug mode used for debugging the code.
        FAST_DEV_RUN (str): Represents the fast development run mode used for quick testing.
        TRAIN (str): Represents the train mode used for training the model.
    """

    DEBUG = 'debug'
    FAST_DEV_RUN = 'fast_dev_run'
    TRAIN = 'train'


class Accelerators(StrEnum):
    """
    Enum for accelerator types.

    Attributes:
        AUTO (str): Represents the automatic selection of accelerator based on availability.
        CPU (str): Represents the Central Processing Unit as the accelerator.
        GPU (str): Represents the Graphics Processing Unit as the accelerator.
    """

    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'


class Fields(StrEnum):
    """
    Enum for field names in the data.

    Attributes:
        BATCH (str): Represents the article_batch.
        PARAGRAPH_ID (str): Represents the paragraph_id.
        UNIQUE_PARAGRAPH_ID (str): Represents the unique_paragraph_id.
        ARTICLE_ID (str): Represents the article_id.
        ARTICLE_IND (str): Represents the article_index.
        QUESTION (str): Represents the question.
        LEVEL (str): Represents the difficulty_level.
        LIST (str): Represents the list_number.
        PARAGRAPH (str): Represents the paragraph.
        HAS_PREVIEW (str): Represents the question_preview.
        SUBJECT_ID (str): Represents the participant_id.
        FINAL_ANSWER (str): Represents the selected_answer_position.
        REREAD (str): Represents the repeated_reading_trial.
        IA_DATA_IA_ID_COL_NAME (str): Represents the IA_ID.
        FIXATION_REPORT_IA_ID_COL_NAME (str): Represents the CURRENT_FIX_INTEREST_AREA_INDEX.
        IS_CORRECT (str): Represents the is_correct.
        PRACTICE (str): Represents the practice_trial.
    """

    UNIQUE_TRIAL_ID = 'unique_trial_id'
    BATCH = 'article_batch'
    PARAGRAPH_ID = 'paragraph_id'
    UNIQUE_PARAGRAPH_ID = 'unique_paragraph_id'
    ARTICLE_ID = 'article_id'
    LEVEL = 'difficulty_level'
    LIST = 'list_number'
    PARAGRAPH = 'paragraph'
    HAS_PREVIEW = 'question_preview'
    SUBJECT_ID = 'participant_id'
    REREAD = 'repeated_reading_trial'
    IA_DATA_IA_ID_COL_NAME = 'IA_ID'
    FIXATION_REPORT_IA_ID_COL_NAME = 'CURRENT_FIX_INTEREST_AREA_INDEX'
    IS_CORRECT = 'is_correct'
    PRACTICE = 'practice_trial'
    QUESTION = 'question'


class ItemLevelFeaturesModes(StrEnum):
    """
    Enum for item-level feature modes.

    Attributes:
    """

    RF = 'RF'
    BEYELSTM = 'BEYELSTM'
    SVM = 'SVM'
    LOGISTIC = 'LOGISTIC'
    READING_SPEED = 'READING_SPEED'


class BackboneNames(StrEnum):
    """
    Enum for backbone names.

    Attributes:
        ROBERTA_BASE (str): Represents the base version of the RoBERTa model.
        ROBERTA_LARGE (str): Represents the large version of the RoBERTa model.
        ROBERTA_RACE (str): Represents the fine-tuned-on-RACE RoBERTa model.
    """

    ROBERTA_BASE = 'roberta-base'
    ROBERTA_LARGE = 'roberta-large'
    ROBERTA_RACE = 'LIAMF-USP/roberta-large-finetuned-race'
    XLM_ROBERTA_BASE = 'FacebookAI/xlm-roberta-base'
    XLM_ROBERTA_LARGE = 'FacebookAI/xlm-roberta-large'


class DLModelNames(StrEnum):
    """
    Enum for model names.

    Attributes:
        MAG_MODEL (str): Represents the name of the MAG model.
        ROBERTEYE_MODEL (str): Represents the name of the Eye BERT model.
        AHN_CNN_MODEL (str): Represents the name of the AHN CNN model.
        AHN_RNN_MODEL (str): Represents the name of the AHN RNN model.
        BEYELSTM_MODEL (str): Represents the name of the BEYELSTM model.
        POSTFUSION_MODEL (str): Represents the name of the PostFusion model.
        PLMAS_MODEL (str): Represents the name of the PLMAS model.
        PLMASF_MODEL (str): Represents the name of the PLMASF model.
    """

    ROBERTEYE_MODEL = 'Roberteye'
    POSTFUSION_MODEL = 'PostFusionModel'
    PLMAS_MODEL = 'PLMASModel'
    PLMASF_MODEL = 'PLMASFModel'
    MAG_MODEL = 'MAGModel'
    AHN_CNN_MODEL = 'AhnCNNModel'
    AHN_RNN_MODEL = 'AhnRNNModel'
    BEYELSTM_MODEL = 'BEyeLSTMModel'


class MLModelNames(StrEnum):
    """
    Enum for ML model names.

    Attributes:
        LOGISTIC_REGRESSION (str): Represents the logistic regression model.
        SVM (str): Represents the support vector machine model.
        RANDOM_FOREST (str): Represents the random forest model.
        DUMMY_CLASSIFIER (str): Represents the dummy classifier model.
        XGBOOST (str): Represents the XGBoost model.
        LOGISTIC_REGRESSION_REG (str): Represents the logistic regression regressor.
        SVM_REG (str): Represents the support vector regressor.
        RANDOM_FOREST_REG (str): Represents the random forest regressor.
        DUMMY_REGRESSOR (str): Represents the dummy regressor.
        XGBOOST_REG (str): Represents the XGBoost regressor.
    """

    LOGISTIC_REGRESSION = 'LogisticRegressionMLModel'
    SVM = 'SupportVectorMachineMLModel'
    RANDOM_FOREST = 'RandomForestMLModel'
    DUMMY_CLASSIFIER = 'DummyClassifierMLModel'
    XGBOOST = 'XGBoostMLModel'
    LINEAR_REG = 'LinearRegressionRegressorMLModel'
    SVM_REG = 'SupportVectorRegressorMLModel'
    RANDOM_FOREST_REG = 'RandomForestRegressorMLModel'
    DUMMY_REGRESSOR = 'DummyRegressorMLModel'
    XGBOOST_REG = 'XGBoostRegressorMLModel'


class TaskTypes(StrEnum):
    """
    Enum for task types.

    Attributes:
        BINARY_CLASSIFICATION (str): Represents binary classification tasks.
        REGRESSION (str): Represents regression tasks.
    """

    BINARY_CLASSIFICATION = 'binary_classification'
    REGRESSION = 'regression'


class PredMode(StrEnum):
    """
    Enum for prediction modes.
    """

    RCS = 'RCS'  # Reading Comprehension Skill
    RC = 'RC'  # Reading Comprehension
    STD = 'STD'  # Subjective Text Difficulty
    TYP = 'TYP'  # Typicality
    CV = 'CV'  # Claim Verification
    LEX = 'LEX'  # Vocabulary Knowledge
    DE = 'DE'  # Domain Expertise


BINARY_PARAGRAPH_ONLY_TASKS = [
    PredMode.RCS,
    PredMode.TYP,
    PredMode.CV,
    PredMode.DE,
]
BINARY_P_AND_Q_TASKS = [PredMode.RC]
REGRESSION_PARAGRAPH_ONLY_TASKS = [PredMode.RCS, PredMode.LEX, PredMode.STD]


class Precision(StrEnum):
    """
    Enum for precision types.

    Attributes:
        SIXTEEN_MIXED (str): Corresponds to "16-mixed".
        THIRTY_TWO_TRUE (str): Corresponds to "32-true".
    """

    SIXTEEN_MIXED = '16-mixed'
    THIRTY_TWO_TRUE = '32-true'


class MatmulPrecisionLevel(StrEnum):
    """
    Enum for matrix multiplication precision levels.

    Attributes:
        HIGHEST (str): Corresponds to "highest".
        HIGH (str): Corresponds to "high".
        MEDIUM (str): Corresponds to "medium".
    """

    HIGHEST = 'highest'
    HIGH = 'high'
    MEDIUM = 'medium'


class Scaler(Enum):
    """
    Enum for scaler types. Each scaler type is associated with a
        corresponding scaler class from sklearn.preprocessing.

    Attributes:
        MIN_MAX_SCALER (type): Corresponds to sklearn.preprocessing.MinMaxScaler.
        ROBUST_SCALER (type): Corresponds to sklearn.preprocessing.RobustScaler.
        STANDARD_SCALER (type): Corresponds to sklearn.preprocessing.StandardScaler.
    """

    MIN_MAX_SCALER = MinMaxScaler
    ROBUST_SCALER = RobustScaler
    STANDARD_SCALER = StandardScaler


class ConfigName(StrEnum):
    """
    Enum for config names.

    Attributes:
        DATA (str): Represents the data config.
        TRAINER (str): Represents the trainer config.
        MODEL (str): Represents the model config.
    """

    DATA = 'data'
    TRAINER = 'trainer'
    MODEL = 'model'


class FeatureMode(StrEnum):
    """
    Enum for feature modes.

    Attributes:
        EYES (str): Represents the eyes feature mode.
        WORDS (str): Represents the words feature mode.
        EYES_WORDS (str): Represents the combined eyes and words feature mode.
    """

    EYES = 'eyes'
    WORDS = 'words'
    EYES_WORDS = 'eyes_words'


class SetNames(StrEnum):
    """
    Enum for set names.

    Attributes:
        TRAIN (str): Represents the training set.
        VAL (str): Represents the validation set.
        TEST (str): Represents the test set.
        SEEN_SUBJECT_UNSEEN_ITEM (str): Represents the seen subject unseen item set.
        UNSEEN_SUBJECT_SEEN_ITEM (str): Represents the unseen subject seen item set.
        UNSEEN_SUBJECT_UNSEEN_ITEM (str): Represents the unseen subject unseen item set.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    SEEN_SUBJECT_UNSEEN_ITEM = 'seen_subject_unseen_item'
    UNSEEN_SUBJECT_SEEN_ITEM = 'unseen_subject_seen_item'
    UNSEEN_SUBJECT_UNSEEN_ITEM = 'unseen_subject_unseen_item'


REGIMES = [
    SetNames.SEEN_SUBJECT_UNSEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_SEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM,
]


class DataType(StrEnum):
    """
    DataType is an enumeration that represents different types of data used in the application.

    Attributes:
        IA (str): Represents interest area data.
        FIXATIONS (str): Represents fixation data.
        RAW (str): Represents raw eye-tracking data.
        TRIAL_LEVEL (str): Represents trial-level aggregated data.
        METADATA (str): Represents metadata.
    """

    IA = 'ia'
    FIXATIONS = 'fixations'
    RAW = 'raw'
    TRIAL_LEVEL = 'trial_level'
    METADATA = 'metadata'


class DataSets(StrEnum):
    """
    DataSets is an enumeration that represents different datasets used in the application.

    Attributes:
        ONESTOP (str): Represents the OneStop dataset.
        COPCO (str): Represents the CopCo dataset.
        POTEC (str): Represents the PoTeC dataset.
        SBSAT (str): Represents the SBSAT dataset.
        HALLUCINATION (str): Represents the IITBHGC dataset.
        MECO_L2 (str): Represents the MECO L2 dataset.
        MECO_L2W1 (str): Represents the MECO L2W1 dataset.
        MECO_L2W2 (str): Represents the MECO L2W2 dataset.
    """

    ONESTOP = 'OneStop'
    COPCO = 'CopCo'
    POTEC = 'PoTeC'
    SBSAT = 'SBSAT'
    HALLUCINATION = 'IITBHGC'
    MECO_L2 = 'MECOL2'
    MECO_L2W1 = 'MECOL2W1'
    MECO_L2W2 = 'MECOL2W2'


class DiscriSupportedMetrics(StrEnum):
    BALANCED_ACCURACY = 'balanced_accuracy'
    AUROC = 'auroc'


class RegrSupportedMetrics(StrEnum):
    RMSE = 'rmse'
    MAE = 'mae'
    R2 = 'r2'
