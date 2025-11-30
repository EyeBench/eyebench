from dataclasses import dataclass

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class PostFusion(DLModelArgs):
    """
    PostFusion Model

    Attributes:
        batch_size (int): Batch size for training.
        accumulate_grad_batches (int): Number of batches to accumulate gradients before updating the weights.
        backbone (BackboneNames): Backbone model to use.
        use_fixation_report (bool): Whether to use fixation report.
        sep_token_id (int): ID of the separator token.
        is_training (bool): Whether the model is in training mode.
        freeze (bool): Whether to freeze the model.
        prepend_eye_features_to_text (bool): A flag indicating whether to prepend the eye data to the input.
            If True, the eye data will be added at the beginning of the input (otherwise no eyes).
        eye_projection_dropout (float): Dropout rate for the eye projection layer.
        cross_attention_dropout (float): Dropout rate for the cross-attention layer.
        use_attn_mask (bool): Whether to use an attention mask in the model.
    """

    base_model_name: DLModelNames = DLModelNames.POSTFUSION_MODEL

    prepend_eye_features_to_text: bool = False
    eye_projection_dropout: float = 0.1
    cross_attention_dropout: float = 0.1
    use_attn_mask: bool = True
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = True
    sep_token_id: int = 2
    is_training: bool = False
    freeze: bool = False
    warmup_proportion: float = 0.1
    max_epochs: int = 10
    early_stopping_patience: int = 3
