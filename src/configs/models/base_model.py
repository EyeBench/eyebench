"""
This module contains dataclasses for defining model arguments and parameters.

The module defines a hierarchy of configuration classes:
- Common base configuration classes for all models (BaseModelParams, CommonBaseModelArgs)
- Specialized configuration classes for deep learning models (BaseModelArgs)
- Specialized configuration classes for machine learning models (MLModelArgs)
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from omegaconf import MISSING

from src.configs.constants import (
    BackboneNames,
    DLModelNames,
    FeatureMode,
    ItemLevelFeaturesModes,
    MLModelNames,
    NormalizationModes,
    Scaler,
)


@dataclass
class BaseModelArgs:
    """
    Common base configuration class for model arguments shared by all models.

    This class contains the shared parameters between deep learning and machine learning models.

    Attributes:
        batch_size (int): The batch size for training.
        text_dim (int): The dimension of the text input.
        max_supported_seq_len (int): The maximum sequence length for the eye input, in tokens.

        use_fixation_report (bool): Whether to use fixation report.
        eyes_dim (int | None): The dimension of the eye features. Defined according to ia_features.
        fixation_dim (int | None): The dimension of the fixation features.
            Defined according to fixation_features + ia_features
        feature_mode (FeatureMode): The mode for features.
        word_features (list[str]): List of word features.
        eye_features (list[str]): List of eye features.
        ia_features (list[str]): List of item-level features.
        fixation_features (list[str]): List of fixation features.
        ia_categorical_features (list[str]): List of categorical features.
        compute_trial_level_features (bool): Whether to compute trial-level features.
        n_tokens (int): Number of tokens.
        eye_token_id (int): Eye token ID.
        sep_token_id (int): Separator token ID.
        is_training (bool): Whether the model is in training mode.
        normalization_mode (NormalizationModes): Mode for normalization.
        normalization_type (Scaler): Type of scaler for normalization.
        class_weights: Weights for each class in the loss function.
            None means equal weight for all classes. Default is None.
        prepend_eye_features_to_text: A flag indicating whether to prepend the eye data to the input.
            If True, the eye data will be added at the beginning of the input. Default is False.
        item_level_features_modes: Modes for item-level features.
        is_ml (bool): Whether the model is a machine learning model or deep learning model.
    """

    use_class_weighted_loss: bool = True
    class_weights: list[float] | None = None  # if use_class_weighted_loss else None
    prepend_eye_features_to_text: bool = False
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=list
    )

    batch_size: int = MISSING
    text_dim: int = -1
    # query to filter longest: (participant_id != 'l31_388' | unique_paragraph_id != '3_1_Adv_4')
    use_fixation_report: bool = MISSING
    max_tokens_in_word: int = 15
    eyes_dim: int = MISSING
    fixation_dim: int = MISSING
    use_eyes_only: bool = False
    is_ml: bool = False
    num_special_tokens_add: int = MISSING

    feature_mode: FeatureMode = FeatureMode.EYES_WORDS
    word_features: list[str] = field(
        default_factory=lambda: [
            'gpt2_surprisal',
            'wordfreq_frequency',
            'word_length',
            'start_of_line',
            'end_of_line',
            'is_content_word',
            'ptb_pos',
            'left_dependents_count',
            'right_dependents_count',
            'distance_to_head',
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            'IA_DWELL_TIME',
            'IA_DWELL_TIME_%',
            'IA_FIXATION_%',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_RUN_COUNT',
            'IA_FIRST_FIXATION_DURATION',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_FIRST_RUN_DWELL_TIME',
            'IA_FIRST_RUN_FIXATION_COUNT',
            'IA_SKIP',
            'IA_REGRESSION_PATH_DURATION',
            'IA_REGRESSION_OUT_COUNT',
            'IA_SELECTIVE_REGRESSION_PATH_DURATION',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_TOP',
            'IA_LEFT',
            'IA_FIRST_FIX_PROGRESSIVE',
            'normalized_ID',
            'PARAGRAPH_RT',
            'total_skip',
        ]
    )

    ia_features: list[str] = MISSING

    fixation_features: list[str] = field(
        default_factory=lambda: [
            'CURRENT_FIX_INDEX',
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_PUPIL',
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y',
            'NEXT_FIX_ANGLE',
            'CURRENT_FIX_INTEREST_AREA_INDEX',
            'NEXT_FIX_INTEREST_AREA_INDEX',
            'PREVIOUS_FIX_ANGLE',
            'NEXT_FIX_DISTANCE',
            'PREVIOUS_FIX_DISTANCE',
            'NEXT_SAC_AMPLITUDE',
            'NEXT_SAC_ANGLE',
            'NEXT_SAC_AVG_VELOCITY',
            'NEXT_SAC_DURATION',
            'NEXT_SAC_PEAK_VELOCITY',
        ]
    )

    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            'ptb_pos',
        ]
    )

    compute_trial_level_features: bool = False
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False
    full_model_name: str = ''
    max_time_limit: str | None = None
    sweep_hours_limit: int = 120
    base_model_name: DLModelNames | MLModelNames = MISSING
    normalization_mode: NormalizationModes = NormalizationModes.ALL
    normalization_type: Scaler = Scaler.ROBUST_SCALER
    backbone: BackboneNames | None = None
    max_supported_seq_len: int = 512  # for roberta-based models

    def __post_init__(self):
        """
        Post-initialization hook to compute `eyes_dim` and `fixation_dim` based on the features.
        """
        if self.feature_mode == FeatureMode.EYES_WORDS:
            self.ia_features = self.eye_features + self.word_features
        elif self.feature_mode == FeatureMode.EYES:
            self.ia_features = self.eye_features
        elif self.feature_mode == FeatureMode.WORDS:
            self.ia_features = self.word_features

        n_categorical_features = len(self.ia_categorical_features)
        assert len(self.ia_features) >= n_categorical_features, (
            'ia_features should be greater or equal to than ia_categorical_features'
        )
        self.eyes_dim = len(self.ia_features) - n_categorical_features
        self.fixation_dim = len(self.fixation_features) + self.eyes_dim

        self.ia_features_to_add_to_fixation_data = self.ia_features

        self.num_special_tokens_add = 6 if self.prepend_eye_features_to_text else 4

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @property
    def max_time(self) -> str | None:
        """
        Returns the maximum time for training in the format of hours:minutes:seconds.
        """
        # Lazy import to avoid circular dependency
        from src.run.multi_run.search_spaces import search_space_by_model
        from src.run.multi_run.utils import count_hyperparameter_configs

        # Check if the model name is in the search space
        if self.base_model_name not in search_space_by_model:
            logger.warning(
                f'Model name {self.base_model_name} not found in search space.'
            )
            return None

        max_time_limit, _ = count_hyperparameter_configs(
            search_space_by_model[self.base_model_name],
            log_specific_values=False,
            n_hours=self.sweep_hours_limit,
        )

        return max_time_limit

    @staticmethod
    def get_text_dim(backbone: BackboneNames | None) -> int:
        """
        Get the text dimension based on the backbone model.

        Args:
            backbone (BackboneNames | str): The backbone model name.

        Returns:
            int: The text dimension.
        """
        if not backbone:
            # print('Backbone is None. Setting text_dim to 0')
            return 0
        if backbone in (BackboneNames.ROBERTA_BASE, BackboneNames.XLM_ROBERTA_BASE):
            # print(f'Backbone: {backbone}. Setting text_dim to 768')
            return 768
        if backbone in (BackboneNames.ROBERTA_LARGE, BackboneNames.XLM_ROBERTA_LARGE):
            # print(f'Backbone {backbone} is recognized, setting text_dim to 1024')
            return 1024
        else:
            # print(f'Backbone {backbone} not recognized, setting text_dim to 0')
            return 0


@dataclass
class DLModelArgs(BaseModelArgs):
    """
    Base configuration class for deep learning model arguments.

    Attributes:
        backbone (BackboneNames): The backbone model to use. Must be specified.
        accumulate_grad_batches (int): The number of batches to accumulate
            gradients before updating the weights.
        hf_access_token (str | None): HuggingFace access token for private models.
        preorder (bool): Order the answers and convert labels
            according to ABCD order before model input.
        warmup_proportion (float | None): Proportion of training steps for learning rate warmup.
            Default is None.
        max_epochs (int | None): Maximum number of training epochs.
        early_stopping_patience (int | None): Number of epochs to wait for improvement before early stopping.

    """

    hf_access_token: str | None = None
    accumulate_grad_batches: int = 1
    preorder: bool = True
    base_model_name: DLModelNames = MISSING
    warmup_proportion: float | None = None
    max_epochs: int | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self):
        """
        Post-initialization hook to compute dimensions and set derived parameters.
        """
        super().__post_init__()
        self.text_dim = self.get_text_dim(self.backbone)


@dataclass
class MLModelArgs(BaseModelArgs):
    """
    Base configuration class for machine learning model arguments.

    Attributes:
        preorder (bool): Whether to preorder the data.
        sklearn_pipeline (Any): The scikit-learn pipeline for the model.
        sklearn_pipeline_params (dict): Parameters for the scikit-learn pipeline.
    """

    base_model_name: MLModelNames = MISSING
    compute_trial_level_features: bool = True
    sklearn_pipeline: Any = MISSING
    sklearn_pipeline_params: dict = field(default_factory=dict)
    max_supported_seq_len: int = 1_000_000

    # Additional features for ML models
    pca_explained_variance_ratio_threshold: float = 1.0  # if 1, no PCA is done

    def init_sklearn_pipeline_params(self):
        """
        Initialize scikit-learn pipeline parameters.
        Iterates over the attributes of the class and adds them to the pipeline parameters.
        """
        # create sklearn pipeline params
        # pass over the attributes of the class and add them to the pipeline params
        for key, value in self.__dict__.items():
            if key.startswith('sklearn_pipeline_param_'):
                self.sklearn_pipeline_params[
                    key.replace('sklearn_pipeline_param_', '')
                ] = value

    preorder: bool = False
    use_eyes_only: bool = True
