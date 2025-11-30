"""
BeyeLSTM model arguments and parameters.
"""

from dataclasses import dataclass, field

from src.configs.constants import (
    DLModelNames,
    ItemLevelFeaturesModes,
)
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class BEyeLSTMArgs(DLModelArgs):
    """
    Model arguments for the BEyeLSTM model.
    #? update the docstring to include all parameters
    Attributes:
        max_eye_len (int): The maximum sequence length for the eye input, in tokens.
        batch_size (int): Batch size for training.
        backbone (BackboneNames): Backbone model to use.
        use_fixation_report (bool): Whether to use fixation report.
        compute_trial_level_features (bool): Whether to compute trial-level features.
        fixation_features (list[str]): List of fixation features.
        eye_features (list[str]): List of eye features.
        word_features (list[str]): List of word features.
        ia_categorical_features (list[str]): List of categorical features for interest areas.
        num_pos (int): Number of positions.
        num_content (int): Number of content types.
        fixations_dim (int): Dimension of fixations.
        gsf_dim (int): Dimension of GSF.
    """

    max_supported_seq_len: int = 1_000_000
    base_model_name: DLModelNames = DLModelNames.BEYELSTM_MODEL
    batch_size: int = 64
    use_fixation_report: bool = True
    use_eyes_only: bool = True
    num_pos: int = 5
    num_content: int = 2
    fixations_dim: int = 4  #! Not a hyperparameter to play with
    """
    Originally:
    **35** binned values X (**13** reading features + **5** linguistic features) + **4** global features = **634**
    Ours:
    **44** binned values X (**11** reading features + **5** linguistic features) + **4** global features = **708**
    """
    gsf_dim: int = -1
    dropout_rate: float = 0.5  # Dropout rate of fc1 and fc2
    embedding_dim: int = (
        4  # The embedding dimension for categorical features (universal_pos, Content)
    )
    # The output dimensions for fc1,2 after each LSTM
    lstm_block_fc1_out_dim: int = 50  # originally: 50
    lstm_block_fc2_out_dim: int = 20  # originally: 20
    gsf_out_dim: int = 32  # originally: 32
    # The middle embedding size of the FC after the concat of all LSTM results and gsf (all separate layers, only the dim is shared)
    after_cat_fc_hidden_dim: int = 32
    hidden_dim: int = 64  # the hidden dim inside the LSTM. Originally: 25

    compute_trial_level_features: bool = True
    fixation_features: list[str] = field(
        default_factory=lambda: [
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_PUPIL',
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y',
            'NEXT_FIX_INTEREST_AREA_INDEX',
            'CURRENT_FIX_INTEREST_AREA_INDEX',
            'LengthCategory',
            'is_reg_sum',
            'is_progressive_sum',
            'IA_REGRESSION_IN_COUNT_sum',
            'normalized_outgoing_regression_count',
            'normalized_outgoing_progressive_count',
            'normalized_incoming_regression_count',
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            'TRIAL_IA_COUNT',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'IA_FIRST_FIXATION_DURATION',
            'IA_DWELL_TIME',
        ]
    )

    word_features: list[str] = field(
        default_factory=lambda: [
            'is_content_word',
            'ptb_pos',
            'left_dependents_count',
            'right_dependents_count',
            'distance_to_head',
            'head_direction',
            'gpt2_surprisal',
            'wordfreq_frequency',
            'word_length',
            # 'entity_type',
            'universal_pos',
        ]
    )

    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            'is_content_word',
            'ptb_pos',
            # 'entity_type',
            'universal_pos',
            'Head_Direction',
            'head_direction',
            'TRIAL_IA_COUNT',
            'LengthCategory',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
        ]
    )

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.BEYELSTM]
    )
    max_epochs: int = 1000
    early_stopping_patience: int = 50
