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
class SupportVectorMachineMLArgs(MLModelArgs):
    """
    Model arguments for the Support Vector Machine (SVM) model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__C (float): Regularization parameter. Inverse of regularization strength.
        sklearn_pipeline_param_clf__kernel (str): Specifies the kernel type to be used in the algorithm.
        sklearn_pipeline_param_clf__degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
        sklearn_pipeline_param_clf__gamma (str | float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        sklearn_pipeline_param_clf__coef0 (float): Independent term in kernel function. Relevant for 'poly' and 'sigmoid'.
        sklearn_pipeline_param_clf__shrinking (bool): Whether to use the shrinking heuristic.
        sklearn_pipeline_param_clf__probability (bool): Whether to enable probability estimates.
        sklearn_pipeline_param_clf__tol (float): Tolerance for stopping criterion.
        sklearn_pipeline_param_clf__random_state (int): Seed for shuffling the data.
        sklearn_pipeline_param_clf__class_weight (str): Class weights (e.g., 'balanced').
        sklearn_pipeline_param_scaler__with_mean (bool): If True, center the data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): If True, scale the data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.SVM

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'sklearn.svm.SVC'),
    )
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__C: float = 100
    sklearn_pipeline_param_clf__kernel: str = 'rbf'
    sklearn_pipeline_param_clf__degree: int = 3  # Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.
    sklearn_pipeline_param_clf__gamma: str | float = 0.01
    # sklearn_pipeline_param_clf__gamma: str = "scale"
    sklearn_pipeline_param_clf__coef0: float = (
        0.0  # It is only significant in ‘poly’ and ‘sigmoid’.
    )
    sklearn_pipeline_param_clf__shrinking: bool = True
    sklearn_pipeline_param_clf__probability: bool = False
    sklearn_pipeline_param_clf__tol: float = 0.001
    sklearn_pipeline_param_clf__random_state: int = 1
    sklearn_pipeline_param_clf__class_weight: str = 'balanced'

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024

    #! note logistic regression is for binary classification
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.SVM]
    )


@register_model_config
@dataclass
class SupportVectorRegressorMLArgs(MLModelArgs):
    """
    Model arguments for the Support Vector Regressor (SVR) model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_reg__C (float): Regularization parameter.
        sklearn_pipeline_param_reg__kernel (str): Specifies the kernel type to be used in the algorithm.
        sklearn_pipeline_param_reg__degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
        sklearn_pipeline_param_reg__gamma (str | float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        sklearn_pipeline_param_reg__coef0 (float): Independent term in kernel function. Relevant for 'poly' and 'sigmoid'.
        sklearn_pipeline_param_reg__shrinking (bool): Whether to use the shrinking heuristic.
        sklearn_pipeline_param_reg__tol (float): Tolerance for stopping criterion.
        sklearn_pipeline_param_reg__epsilon (float): Epsilon in the epsilon-SVR model.
        sklearn_pipeline_param_scaler__with_mean (bool): If True, center the data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): If True, scale the data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.SVM_REG

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('reg', 'sklearn.svm.SVR'),
    )

    # regressor params
    sklearn_pipeline_param_reg__C: float = 100
    sklearn_pipeline_param_reg__kernel: str = 'rbf'
    sklearn_pipeline_param_reg__degree: int = 3
    sklearn_pipeline_param_reg__gamma: str | float = 0.01
    sklearn_pipeline_param_reg__coef0: float = 0.0
    sklearn_pipeline_param_reg__shrinking: bool = True
    sklearn_pipeline_param_reg__tol: float = 0.001
    sklearn_pipeline_param_reg__epsilon: float = 0.1

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.SVM]
    )
