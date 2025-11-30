from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ItemLevelFeaturesModes,
    MLModelNames,
)
from src.configs.models.base_model import (
    MLModelArgs,
)
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class XGBoostMLArgs(MLModelArgs):
    """
    Model arguments for the XGBoost model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        pca_explained_variance_ratio_threshold (float): Threshold for PCA explained variance ratio.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__learning_rate (float): Learning rate for the XGBoost model.
        sklearn_pipeline_param_clf__min_child_weight (int): Minimum sum of instance weight (hessian) needed in a child.
        sklearn_pipeline_param_clf__gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
        sklearn_pipeline_param_clf__n_estimators (int): Number of gradient boosted trees.
        sklearn_pipeline_param_clf__max_depth (int): Maximum depth of a tree.
        sklearn_pipeline_param_clf__colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        sklearn_pipeline_param_clf__alpha (float): L1 regularization term on weights.
        sklearn_pipeline_param_clf__lambda (float): L2 regularization term on weights.
        sklearn_pipeline_param_clf__booster (str): Type of booster to use.
        sklearn_pipeline_params_clf__device (str): Device to use for training (e.g., "gpu").
        sklearn_pipeline_param_scaler__with_mean (bool): If True, center the data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): If True, scale the data to unit variance (or equivalently, unit standard deviation).
    """

    base_model_name: MLModelNames = MLModelNames.XGBOOST

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'xgboost.XGBClassifier'),
    )

    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__learning_rate: float = 0.01
    sklearn_pipeline_param_clf__min_child_weight: int = 1
    sklearn_pipeline_param_clf__gamma: float = 0
    sklearn_pipeline_param_clf__n_estimators: int = 1000
    sklearn_pipeline_param_clf__max_depth: int = 6
    sklearn_pipeline_param_clf__colsample_bytree: float = 1.0
    sklearn_pipeline_param_clf__alpha: float = 0
    sklearn_pipeline_param_clf__lambda: float = 1
    sklearn_pipeline_param_clf__booster: str = 'gbtree'

    # sklearn_pipeline_param_clf__scale_pos_weight: float = sqrt(
    #     83.6 / 16.4
    # )  # the ratio between 0 and 1 in the reread column of the train set of fold 0
    sklearn_pipeline_params_clf__device: str = 'gpu'
    # sklearn_pipeline_param_clf__shrinking: bool = True
    # sklearn_pipeline_param_clf__probability: bool = False
    # sklearn_pipeline_param_clf__tol: float = 0.001
    # sklearn_pipeline_param_clf__random_state: int = 1
    # sklearn_pipeline_param_clf__class_weight: str = "balanced"

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024

    #! note logistic regression is for binary classification
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.RF],
    )


@register_model_config
@dataclass
class XGBoostRegressorMLArgs(MLModelArgs):
    """
    Model arguments for the XGBoost regressor model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        pca_explained_variance_ratio_threshold (float): Threshold for PCA explained variance ratio.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_reg__learning_rate (float): Learning rate for the XGBoost model.
        sklearn_pipeline_param_reg__min_child_weight (int): Minimum sum of instance weight (hessian) needed in a child.
        sklearn_pipeline_param_reg__gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
        sklearn_pipeline_param_reg__n_estimators (int): Number of gradient boosted trees.
        sklearn_pipeline_param_reg__max_depth (int): Maximum depth of a tree.
        sklearn_pipeline_param_reg__colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        sklearn_pipeline_param_reg__alpha (float): L1 regularization term on weights.
        sklearn_pipeline_param_reg__lambda (float): L2 regularization term on weights.
        sklearn_pipeline_param_reg__booster (str): Type of booster to use.
        sklearn_pipeline_params_reg__device (str): Device to use for training (e.g., "gpu").
        sklearn_pipeline_param_scaler__with_mean (bool): If True, center the data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): If True, scale the data to unit variance (or equivalently, unit standard deviation).
    """

    base_model_name: MLModelNames = MLModelNames.XGBOOST_REG

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('reg', 'xgboost.XGBRegressor'),
    )

    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # regressor params
    sklearn_pipeline_param_reg__learning_rate: float = 0.01
    sklearn_pipeline_param_reg__min_child_weight: int = 1
    sklearn_pipeline_param_reg__gamma: float = 0
    sklearn_pipeline_param_reg__n_estimators: int = 1000
    sklearn_pipeline_param_reg__max_depth: int = 6
    sklearn_pipeline_param_reg__colsample_bytree: float = 1.0
    sklearn_pipeline_param_reg__alpha: float = 0
    sklearn_pipeline_param_reg__lambda: float = 1
    sklearn_pipeline_param_reg__booster: str = 'gbtree'
    sklearn_pipeline_params_reg__device: str = 'gpu'

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024

    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.RF],
    )
