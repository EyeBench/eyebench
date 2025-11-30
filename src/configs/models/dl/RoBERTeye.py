from dataclasses import dataclass

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@dataclass
class RoberteyeArgs(DLModelArgs):
    """
    prepend_eye_features_to_text (bool): A flag indicating whether to prepend the eye data to the input.
        If True, the eye data will be added at the beginning of the input; otherwise, eye data is not used.
    """

    base_model_name: DLModelNames = DLModelNames.ROBERTEYE_MODEL

    prepend_eye_features_to_text: bool = True
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    freeze: bool = False
    eye_projection_dropout: float = 0.3
    max_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_proportion: float = 0.1
    eye_projection_MAGModule: bool = False

    token_type_num: int = 2
    vocab_size: int = -1  # specified in instantiate_config
    #! Don't change the following
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False


@register_model_config
@dataclass
class RoberteyeFixation(RoberteyeArgs):
    """
    Fixation-level RoBERTeye
    """

    use_fixation_report: bool = True
    batch_size: int = 2
    accumulate_grad_batches: int = 16 // batch_size


@register_model_config
@dataclass
class RoberteyeWord(RoberteyeArgs):
    """
    Word-level RoBERTeye
    """

    use_fixation_report: bool = False


@register_model_config
@dataclass
class Roberta(RoberteyeArgs):
    """
    Roberta Model (no eye data)
    """

    prepend_eye_features_to_text: bool = False
    use_fixation_report: bool = False
