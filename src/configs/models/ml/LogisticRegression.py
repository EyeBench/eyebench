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
class LogisticRegressionMLArgs(MLModelArgs):
    """
    Model arguments for the Logistic Regression model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__C (float): Inverse of regularization strength.
        sklearn_pipeline_param_clf__fit_intercept (bool): Whether to add an intercept to the decision function.
        sklearn_pipeline_param_clf__penalty (str): Norm used in penalization.
        sklearn_pipeline_param_clf__solver (str): Optimization algorithm.
        sklearn_pipeline_param_clf__random_state (int): Seed for pseudo-random number generator.
        sklearn_pipeline_param_clf__max_iter (int): Maximum number of solver iterations.
        sklearn_pipeline_param_clf__class_weight (str): Class weight balancing strategy.
        sklearn_pipeline_param_scaler__with_mean (bool): Whether to center data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): Whether to scale data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.LOGISTIC_REGRESSION

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'sklearn.linear_model.LogisticRegression'),
    )
    sklearn_pipeline_param_clf__C: float = 2.0
    sklearn_pipeline_param_clf__fit_intercept: bool = True
    sklearn_pipeline_param_clf__penalty: str = 'l2'
    sklearn_pipeline_param_clf__solver: str = 'lbfgs'
    sklearn_pipeline_param_clf__random_state: int = 1
    sklearn_pipeline_param_clf__max_iter: int = 1000
    sklearn_pipeline_param_clf__class_weight: str = 'balanced'
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED],
    )


@register_model_config
@dataclass
class LogisticMeziereArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )


@register_model_config
@dataclass
class LinearRegressionArgs(MLModelArgs):
    """
    Model arguments for the Linear Regression model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_regressor__fit_intercept (bool): Whether to calculate the intercept for this model.
        sklearn_pipeline_param_scaler__with_mean (bool): Whether to center data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): Whether to scale data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.LINEAR_REG

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('regressor', 'sklearn.linear_model.LinearRegression'),
    )
    sklearn_pipeline_param_regressor__fit_intercept: bool = True
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.XLM_ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED],
    )


@register_model_config
@dataclass
class LinearMeziereArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )
