from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ItemLevelFeaturesModes,
    MLModelNames,
)
from src.configs.models.base_model import MLModelArgs
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class DummyClassifierMLArgs(MLModelArgs):
    """
    Model arguments for the Dummy Classifier model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        item_level_features_modes (list[ItemLevelFeaturesModes]): The item-level features to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__strategy (str): Strategy for the dummy classifier
            ("stratified", "most_frequent", etc.).
        sklearn_pipeline_param_clf__random_state (int): Random seed for the dummy classifier.
    """

    base_model_name: MLModelNames = MLModelNames.DUMMY_CLASSIFIER
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED]
    )
    sklearn_pipeline: tuple = (('clf', 'sklearn.dummy.DummyClassifier'),)
    sklearn_pipeline_param_clf__strategy: str = 'most_frequent'
    sklearn_pipeline_param_clf__random_state: int = 1

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE


@register_model_config
@dataclass
class DummyRegressorMLArgs(MLModelArgs):
    """
    Model arguments for the Dummy Regressor model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        item_level_features_modes (list[ItemLevelFeaturesModes]): The item-level features to use.
        sklearn_pipeline (list): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_reg__strategy (str): Strategy for the dummy regressor ("mean", "median", etc.).
        sklearn_pipeline_param_reg__random_state (int): Random seed for the dummy regressor.
    """

    base_model_name: MLModelNames = MLModelNames.DUMMY_REGRESSOR
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED]
    )
    sklearn_pipeline: tuple = (('reg', 'sklearn.dummy.DummyRegressor'),)
    sklearn_pipeline_param_reg__strategy: str = 'mean'

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
