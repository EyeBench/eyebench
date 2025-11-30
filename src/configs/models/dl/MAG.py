from dataclasses import dataclass

from src.configs.constants import BackboneNames, DLModelNames
from src.configs.models.base_model import DLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class MAG(DLModelArgs):
    """
    Model arguments for the MAG model.

    Attributes:
        batch_size (int): Batch size for training.
        accumulate_grad_batches (int): Number of batches to accumulate gradients over.
        backbone (BackboneNames): Backbone model to use.
        use_fixation_report (bool): Whether to use fixation report.
        freeze (bool): Whether to freeze the model parameters.
        Attributes:
        mag_dropout (float): Dropout rate for the MAG module.
        mag_beta_shift (float): Beta shift parameter used in the MAG module.
        mag_injection_index (int): Index at which the MAG features are injected into the model.
    """

    base_model_name: DLModelNames = DLModelNames.MAG_MODEL

    mag_dropout: float = 0.5
    mag_beta_shift: float = 1e-3
    mag_injection_index: int = 0
    warmup_proportion: float = 0.1
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = False
    freeze: bool = False

    max_epochs: int = 10
    early_stopping_patience: int = 3

    def __post_init__(self):
        super().__post_init__()
        if (
            self.backbone == BackboneNames.ROBERTA_BASE
            and self.mag_injection_index > 13
        ):
            raise ValueError(
                f'Warning: MAG injection index {self.mag_injection_index} is higher than 13 for Roberta Base. Exiting.'
            )
