from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class PLMASfArgs(DLModelArgs):
    """
    Model arguments for the PLMASf model.

    Attributes:
        batch_size (int): The batch size for training.
        accumulate_grad_batches (int): The number of batches to accumulate gradients.
        backbone (BackboneNames): The backbone model to use.
        use_fixation_report (bool): Whether to use fixation report.
        freeze (bool): Whether to freeze the model parameters.
        fixation_features (list[str]): List of fixation features to use.
        eye_features (list[str]): List of eye features to use.
        ia_categorical_features (list[str]): List of categorical interest area features.
        lstm_hidden_size (int): Hidden size for the LSTM layers.
        lstm_num_layers (int): Number of LSTM layers in the model.
        lstm_dropout (float): Dropout rate for the LSTM layers.
    """

    base_model_name: DLModelNames = DLModelNames.PLMASF_MODEL

    lstm_hidden_size: int = 70
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.1

    batch_size: int = 16
    accumulate_grad_batches: int = 1
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = True
    freeze: bool = True

    fixation_features: list[str] = field(
        default_factory=lambda: [
            'CURRENT_FIX_INTEREST_AREA_INDEX',
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
            'CURRENT_FIX_Y',
            'NEXT_SAC_DURATION',
            'NEXT_SAC_END_X',
            'NEXT_SAC_START_X',
            'NEXT_SAC_END_Y',
            'NEXT_SAC_START_Y',
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            'IA_DWELL_TIME',
            # 'IA_FIRST_RUN_LANDING_POSITION',
            # 'IA_LAST_RUN_LANDING_POSITION',
            'IA_FIRST_FIXATION_DURATION',
        ]
    )
    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            'is_content_word',
            'ptb_pos',
            # 'entity_type',
            'universal_pos',
            'Head_Direction',
            'TRIAL_IA_COUNT',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
        ]
    )

    max_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_proportion: float = 0.1
