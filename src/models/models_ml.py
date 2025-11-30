import torch

from src.configs.data import DataArgs
from src.configs.models.ml.DummyClassifier import (
    DummyClassifierMLArgs,
    DummyRegressorMLArgs,
)
from src.configs.models.ml.LogisticRegression import (
    LinearRegressionArgs,
    LogisticRegressionMLArgs,
)
from src.configs.models.ml.RandomForest import (
    RandomForestMLArgs,
    RandomForestRegressorMLArgs,
)
from src.configs.models.ml.SVM import (
    SupportVectorMachineMLArgs,
    SupportVectorRegressorMLArgs,
)
from src.configs.models.ml.XGBoost import XGBoostMLArgs, XGBoostRegressorMLArgs
from src.configs.trainers import TrainerML
from src.models.base_model import BaseMLModel, register_model


@register_model
class LogisticRegressionMLModel(BaseMLModel):
    """
    Logistic Regression classifier.
    """

    def __init__(
        self,
        model_args: LogisticRegressionMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class DummyClassifierMLModel(BaseMLModel):
    """
    Dummy classifier that uses null features and a dummy classifier from sklearn.
    """

    def __init__(
        self,
        model_args: DummyClassifierMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

    def fit(self, dm) -> None:
        # Dummy classifier: generate zero features just for demonstration
        train_batches = self.shared_fit(dm)
        features_list = []
        y_true_list = []
        for train_batch in train_batches:
            features = torch.zeros((train_batch.labels.shape[0], 1)).to('cpu')
            features_list.append(features)
            y_true_list.append(train_batch.labels)

        features = torch.cat(features_list, dim=0).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()
        self.classifier.fit(features, y_true)

    def model_specific_predict(
        self, dev_batches: list
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batches:
            features = torch.zeros((dev_batch.labels.shape[0], 1)).to('cpu').numpy()
            preds, probs = self._predict_with_fallback(features)
            preds_list.append(preds)
            probs_list.append(probs)
        return preds_list, probs_list


@register_model
class SupportVectorMachineMLModel(BaseMLModel):
    """
    Support Vector Machine classifier.
    """

    def __init__(
        self,
        model_args: SupportVectorMachineMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class XGBoostMLModel(BaseMLModel):
    """
    XGBoost classifier.
    """

    def __init__(
        self, model_args: XGBoostMLArgs, trainer_args: TrainerML, data_args=DataArgs
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class RandomForestMLModel(BaseMLModel):
    """
    RandomForest classifier.
    """

    def __init__(
        self,
        model_args: RandomForestMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class LinearRegressionRegressorMLModel(BaseMLModel):
    """
    Logistic Regression classifier.
    """

    def __init__(
        self,
        model_args: LinearRegressionArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class DummyRegressorMLModel(BaseMLModel):
    """
    Dummy regressor that uses null features and a dummy regressor from sklearn.
    """

    def __init__(
        self,
        model_args: DummyRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

    def fit(self, dm) -> None:
        # Dummy classifier: generate zero features just for demonstration
        train_batches = self.shared_fit(dm)
        features_list = []
        y_true_list = []
        for train_batch in train_batches:
            features = torch.zeros((train_batch.labels.shape[0], 1)).to('cpu')
            features_list.append(features)
            y_true_list.append(train_batch.labels)

        features = torch.cat(features_list, dim=0).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()
        self.classifier.fit(features, y_true)

    def model_specific_predict(
        self, dev_batches: list
    ) -> tuple[list[torch.Tensor], list[None]]:
        preds_list: list[torch.Tensor] = []
        probs_list: list[None] = []
        for dev_batch in dev_batches:
            features = torch.zeros((dev_batch.labels.shape[0], 1)).to('cpu').numpy()
            preds = torch.tensor(self.classifier.predict(features))
            probs = None
            preds_list.append(preds)
            probs_list.append(probs)
        return preds_list, probs_list


@register_model
class SupportVectorRegressorMLModel(BaseMLModel):
    """
    Support Vector Machine regressor.
    """

    def __init__(
        self,
        model_args: SupportVectorRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class XGBoostRegressorMLModel(BaseMLModel):
    """
    XGBoost regressor.
    """

    def __init__(
        self,
        model_args: XGBoostRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )


@register_model
class RandomForestRegressorMLModel(BaseMLModel):
    """
    RandomForest Regressor.
    """

    def __init__(
        self,
        model_args: RandomForestRegressorMLArgs,
        trainer_args: TrainerML,
        data_args=DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
