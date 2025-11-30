"""Ahn.py
Ahn model configuration.
This module defines the configuration for the Ahn model, including its parameters and
specific settings for different model architectures (RNN and CNN).
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.configs.constants import DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@dataclass
class Ahn(DLModelArgs):
    """
    Configuration for the Ahn model.
    """

    batch_size: int = 16
    use_fixation_report: bool = True
    use_eyes_only: bool = True
    max_supported_seq_len: int = 1_000_000

    preorder: bool = False

    fixation_features: list[str] = field(
        default_factory=lambda: [
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_PUPIL',
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y',
        ]
    )
    eye_features: list = field(default_factory=list)
    word_features: list = field(default_factory=list)
    ia_categorical_features: list = field(default_factory=list)
    hidden_dim: int = MISSING
    fc_dropout: float = 0.3
    max_epochs: int = 1000
    early_stopping_patience: int = 50


@register_model_config
@dataclass
class AhnRNN(Ahn):
    """
    Configuration for the Ahn RNN model.
    """

    base_model_name: DLModelNames = DLModelNames.AHN_RNN_MODEL

    hidden_dim: int = 25
    num_lstm_layers: int = 1
    fc_hidden_dim: int = 20


@register_model_config
@dataclass
class AhnCNN(Ahn):
    """
    Configuration for the Ahn CNN model.
    """

    base_model_name: DLModelNames = DLModelNames.AHN_CNN_MODEL

    hidden_dim: int = 40
    conv_kernel_size: int = 3
    pooling_kernel_size: int = 2
    fc_hidden_dim1: int = 50
    fc_hidden_dim2: int = 20
