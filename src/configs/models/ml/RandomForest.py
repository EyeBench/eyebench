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
class RandomForestMLArgs(MLModelArgs):
    """
    Model arguments for the RandomForest model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        pca_explained_variance_ratio_threshold (float): Threshold for PCA explained variance ratio.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__n_estimators (int): Number of gradient boosted trees.
        sklearn_pipeline_param_clf__criterion (str): The function to measure the quality of a split.
        sklearn_pipeline_param_clf__max_depth (int): Maximum depth of a tree.
        sklearn_pipeline_param_clf__min_samples_split (int | float): The minimum number of samples required to split an internal node.
        sklearn_pipeline_param_clf__min_samples_leaf (int | float): The minimum number of samples required to be at a leaf node.
        sklearn_pipeline_param_clf__max_features (str| int | float): The number of features to consider when looking for the best split.
        sklearn_pipeline_param_clf__n_jobs (int): The number of jobs to run in parallel.

    """

    base_model_name: MLModelNames = MLModelNames.RANDOM_FOREST

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'sklearn.ensemble.RandomForestClassifier'),
    )

    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__n_estimators: int = 1000
    sklearn_pipeline_param_clf__max_depth: int = 6
    sklearn_pipeline_param_clf__min_samples_split: int = 2
    sklearn_pipeline_param_clf__min_samples_leaf: int = 1
    sklearn_pipeline_param_clf__max_features: str = 'sqrt'
    sklearn_pipeline_param_clf__n_jobs: int = -1

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.RF],
    )


@register_model_config
@dataclass
class RandomForestRegressorMLArgs(MLModelArgs):
    """
    Model arguments for the RandomForest regressor model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        pca_explained_variance_ratio_threshold (float): Threshold for PCA explained variance ratio.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_reg__n_estimators (int): Number of trees in the forest.
        sklearn_pipeline_param_reg__max_depth (int): Maximum depth of a tree.
        sklearn_pipeline_param_reg__min_samples_split (int | float): The minimum number of samples required to split an internal node.
        sklearn_pipeline_param_reg__min_samples_leaf (int | float): The minimum number of samples required to be at a leaf node.
        sklearn_pipeline_param_reg__max_features (str| int | float): The number of features to consider when looking for the best split.
        sklearn_pipeline_param_reg__n_jobs (int): The number of jobs to run in parallel.
    """

    base_model_name: MLModelNames = MLModelNames.RANDOM_FOREST_REG

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('reg', 'sklearn.ensemble.RandomForestRegressor'),
    )

    # sklearn pipeline params
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # regressor params
    sklearn_pipeline_param_reg__n_estimators: int = 1000
    sklearn_pipeline_param_reg__max_depth: int = 6
    sklearn_pipeline_param_reg__min_samples_split: int = 2
    sklearn_pipeline_param_reg__min_samples_leaf: int = 1
    sklearn_pipeline_param_reg__max_features: str = 'sqrt'
    sklearn_pipeline_param_reg__n_jobs: int = -1

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE

    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.RF],
    )
