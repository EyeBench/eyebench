""" This module contains the base model class and the multimodal model class. """ ''

import importlib
import itertools
import warnings
from abc import abstractmethod
from collections import defaultdict, namedtuple
from functools import partial
from typing import Any, Literal, Tuple, Type, Union

import lightning.pytorch as pl
import matplotlib.lines as matplotlib_lines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchmetrics.classification as cls_metrics
import torchmetrics.regression as reg_metrics
import wandb
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.constants import REGIMES, PredMode, SetNames, TaskTypes
from src.configs.data import DataArgs
from src.configs.models.base_model import DLModelArgs, MLModelArgs
from src.configs.trainers import TrainerDL, TrainerML


class SharedBaseModel:
    """Shared base class containing common initialization logic for both DL and ML models."""

    def __init__(
        self, model_args: Union[DLModelArgs, MLModelArgs], data_args: DataArgs
    ):
        """Initialize shared attributes between BaseModel and BaseMLModel.

        Args:
            model_args: Model configuration arguments (either DL or ML)
            data_args: Data configuration arguments
        """
        self.use_eyes_only = model_args.use_eyes_only
        self.use_fixation_report = model_args.use_fixation_report
        self.class_names = list(data_args.class_names)
        self.num_classes = len(self.class_names)
        self.average: Literal['macro'] = 'macro'
        self.validate_metrics: bool = False
        self.prediction_mode: PredMode = data_args.task

        # Determine task type based on number of classes
        if self.num_classes == 1:
            self.task: TaskTypes = TaskTypes.REGRESSION
        elif self.num_classes == 2:
            self.task: TaskTypes = TaskTypes.BINARY_CLASSIFICATION
        elif self.num_classes > 2:
            raise NotImplementedError('Multi-class classification is not implemented.')

        logger.info(f'Using {self.task=} metrics')

        # Handle class weights
        if model_args.use_class_weighted_loss:
            self.class_weights = model_args.class_weights
            logger.info(f'Using class weights: {self.class_weights}')
        else:
            self.class_weights = None
            logger.info('Not using class weights')


class BaseModel(pl.LightningModule, SharedBaseModel):
    """Base model class for the multi-modal models."""

    def __init__(
        self, model_args: DLModelArgs, trainer_args: TrainerDL, data_args: DataArgs
    ):
        # Initialize PyTorch Lightning module
        pl.LightningModule.__init__(self)
        # Initialize shared base attributes
        SharedBaseModel.__init__(self, model_args=model_args, data_args=data_args)
        self.max_data_seq_len = data_args.max_seq_len
        self.max_model_supported_len = model_args.max_supported_seq_len
        self.actual_max_needed_len = min(
            self.max_data_seq_len, self.max_model_supported_len
        )
        # DL-specific attributes
        logger.info(f'{self.use_eyes_only=}')
        self.learning_rate = trainer_args.learning_rate
        self.batch_size = model_args.batch_size
        self.max_scanpath_length: int = data_args.max_scanpath_length

        self.regime_names: list[SetNames] = REGIMES
        logger.info(f'{self.regime_names=}')

        self.val_max_acc_dict = {
            k: {m: 0.0 for m in ['average', 'weighted_average']}
            for k in ['Balanced', 'Classless']
        }

        if self.task == TaskTypes.REGRESSION:
            self.loss = nn.MSELoss()
        else:
            if model_args.use_class_weighted_loss:
                self.loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights))
            else:
                self.loss = nn.CrossEntropyLoss()
        self.validation_step_losses = defaultdict(list)

        (
            metrics,
            confusion_matrix,
            roc,
            balanced_accuracy,
        ) = self.configure_metrics()

        self.train_metrics = metrics.clone(postfix='/train')
        self.val_metrics_list = nn.ModuleList(
            [
                metrics.clone(postfix=f'/val_{regime_name}')
                for regime_name in self.regime_names
            ]
        )
        self.test_metrics_list = nn.ModuleList(
            [
                metrics.clone(postfix=f'/test_{regime_name}')
                for regime_name in self.regime_names
            ]
        )

        if self.task != TaskTypes.REGRESSION:
            self.train_cm = confusion_matrix.clone()
            self.val_cm_list = nn.ModuleList(
                [confusion_matrix.clone() for _ in self.regime_names]
            )
            self.test_cm_list = nn.ModuleList(
                [confusion_matrix.clone() for _ in self.regime_names]
            )
            self.train_roc = roc.clone()
            self.val_roc_list = nn.ModuleList([roc.clone() for _ in self.regime_names])
            self.test_roc_list = nn.ModuleList([roc.clone() for _ in self.regime_names])

            self.train_balanced_accuracy = balanced_accuracy.clone()
            self.val_balanced_accuracy_list = nn.ModuleList(
                [balanced_accuracy.clone() for _ in self.regime_names]
            )
            self.test_balanced_accuracy_list = nn.ModuleList(
                [balanced_accuracy.clone() for _ in self.regime_names]
            )

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def unpack_batch(batch: list) -> tuple:
        """
        Unpacks the batch into a namedtuple for dot access.

        Args:
            batch (list): The batch to unpack containing:
                - batch[0] (dict): Dictionary with features including:
                    - fixation_features: Optional[torch.Tensor] - fixations feature vectors for each fixation.
                    - fixation_pads: Optional[torch.Tensor] - Feature vectors for fixations.
                    - scanpath: Optional[torch.Tensor] - A series of IA_IDs for each trial scanpath.
                    - scanpath_pads: Optional[torch.Tensor] - Feature vectors for IA_IDs in scanpath.
                    - paragraph_input_ids: Optional[torch.Tensor] - Input_ids of each tokenized paragraph.
                    - paragraph_input_masks: Optional[torch.Tensor] - Masks for paragraph input_ids.
                    - input_ids: Optional[torch.Tensor] - Input_ids of tokenized paragraphs.
                    - input_masks: Optional[torch.Tensor] - Masks for input_ids.
                    - answer_mappings: Optional[torch.Tensor] - Mappings associated with answers.
                    - inversions: Optional[torch.Tensor] - For each paragraph, IA_IDs associated with tokens.
                    - inversions_pads: Optional[torch.Tensor] - Pads for IA_IDs in inversions.
                    - eyes: Optional[torch.Tensor] - et_data_enriched feature vectors for each IA_ID.
                - batch[1]: Labels associated with the batch.
                - batch[2]: Keys for each item in the batch.
                - batch[3]: Columns that form a trial.

        Returns:
            tuple: The unpacked batch with dot access (namedtuple).
        """

        ExampleBatch = namedtuple(
            'ExampleBatch',
            list(batch[0].keys())
            + ['labels', 'batch_item_keys', 'trial_groupby_columns'],
        )
        return ExampleBatch(
            **dict(batch[0].items()),
            labels=batch[1],
            batch_item_keys=batch[2],
            trial_groupby_columns=batch[3],
        )

    @abstractmethod
    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def process_step(
        self, batch: list, step_type: str, dataloader_idx=0
    ) -> torch.Tensor:
        labels, loss, logits = self.shared_step(batch)

        metrics = self.get_metrics_map(step_type, dataloader_idx)
        is_single = (labels.ndim == 0) or (logits.ndim == 0)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)

        if self.task == TaskTypes.REGRESSION:
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            if (len(labels) == 1) or (len(logits) == 1):
                is_single = True
            if is_single:
                if logits.ndim != 1:
                    logits = logits.squeeze(0)

                metrics['metrics'].update(logits, labels)
            else:
                metrics['metrics'].update(logits.squeeze(), labels)

        else:
            # Handle last batch with batch size of one (logits may be 1D)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            probs = logits.softmax(dim=1)
            preds = probs.argmax(dim=1)

            metrics['balanced_accuracy'](
                preds, labels
            )  # * must be one entry per class and sample, or after argmax.

            if self.num_classes == 2:
                probs = probs[:, 1]

            metrics['metrics'].update(probs, labels)
            metrics['cm'].update(probs, labels)
            metrics['roc'].update(probs, labels)

        return loss

    def log_loss(self, loss: torch.Tensor, step_type: str, dataloader_idx=0) -> None:
        if step_type == 'train':
            name = 'loss/train'
        else:
            name = f'loss/{step_type}_{self.regime_names[dataloader_idx]}'

        self.log(
            name=name,
            value=loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def training_step(self, batch: list, _) -> torch.Tensor:
        loss = self.process_step(batch, 'train')
        self.log_loss(loss, 'train')
        return loss

    def validation_step(self, batch: list, _, dataloader_idx=0) -> torch.Tensor:
        loss = self.process_step(
            batch=batch, step_type='val', dataloader_idx=dataloader_idx
        )
        self.log_loss(loss, 'val', dataloader_idx)
        self.validation_step_losses[dataloader_idx].append(loss)
        return loss

    def test_step(self, batch: list, _, dataloader_idx=0) -> torch.Tensor:
        loss = self.process_step(
            batch=batch, step_type='test', dataloader_idx=dataloader_idx
        )
        self.log_loss(loss=loss, step_type='test', dataloader_idx=dataloader_idx)
        return loss

    def predict_step(
        self,
        batch: list,
        _,
        dataloader_idx=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            labels,
            unused_loss,
            logits,
        ) = self.shared_step(batch)

        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if self.task == TaskTypes.REGRESSION:
            probs = logits.squeeze()
        else:
            probs = logits.softmax(dim=1)

        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        if self.num_classes == 2:
            probs = probs[:, 1]
        return labels, probs

    def configure_optimizers(self):
        # Define the optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def process_epoch_end(
        self, step_type: str, regime_name: str, index=0
    ) -> tuple[int, torch.Tensor] | None:
        metrics = self.get_metrics_map(step_type=step_type, index=index)

        if self.task == TaskTypes.REGRESSION:
            computed_metrics = metrics['metrics'].compute()
            self.log_dict(
                dictionary=computed_metrics,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                batch_size=self.batch_size,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            metrics['metrics'].reset()
            return

        cm_torch = metrics['cm'].compute()
        cm_formatted = self.format_confusion_matrix(cm=cm_torch)
        self.log_confusion_matrix(
            cm_data=cm_formatted, title=f'ConfusionMatrix/{step_type}_{regime_name}'
        )

        metrics['roc'].compute()
        self.log_roc(roc=metrics['roc'], title=f'ROC/{step_type}_{regime_name}')

        computed_metrics = metrics['metrics'].compute()
        computed_balanced_accuracy = metrics['balanced_accuracy'].compute()
        all_metrics = computed_metrics | {
            f'Balanced_Accuracy/{step_type}_{regime_name}': computed_balanced_accuracy,
        }

        self.log_dict(
            dictionary=all_metrics,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        metrics['roc'].reset()
        metrics['metrics'].reset()
        metrics['cm'].reset()
        metrics['balanced_accuracy'].reset()

        return (
            int(cm_torch.sum().item()),
            computed_balanced_accuracy,
        )  # ::int shouldn't be necessary

    def on_train_epoch_end(self) -> None:
        name_ = 'train'
        self.process_epoch_end(step_type=name_, regime_name=name_)

    def on_validation_epoch_end(self) -> None:
        if self.validation_step_losses:
            all_losses = torch.cat(
                [
                    torch.tensor(losses)
                    for losses in self.validation_step_losses.values()
                ]
            )
            overall_loss = all_losses.mean()
            self.log(
                name='loss/val_all',
                value=overall_loss,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.validation_step_losses.clear()

        if self.task == TaskTypes.REGRESSION:
            return

        # def on_validation_epoch_end(self) -> None:
        counts, balanced_accuracy_values = zip(
            *[
                self.process_epoch_end(
                    step_type='val', index=val_index, regime_name=regime_name
                )
                for val_index, regime_name in enumerate(self.regime_names)
            ]
        )

        balanced_accuracy_values = [x.item() for x in balanced_accuracy_values]

        for k in ['Balanced']:
            accuracy_values = balanced_accuracy_values
            for m in ['average', 'weighted_average']:
                # calculate the current value, update the current and maximum values
                curr_value = (
                    np.average(accuracy_values)
                    if m == 'average'
                    else np.average(accuracy_values, weights=counts)
                )
                assert isinstance(curr_value, float)
                self.val_max_acc_dict[k][m] = max(
                    [self.val_max_acc_dict[k][m], curr_value]
                )

                partial_acc_log = partial(
                    self.log,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )

                partial_acc_log(
                    name=f'{k}_Accuracy/val_{m}',
                    value=curr_value,
                )
                partial_acc_log(
                    name=f'{k}_Accuracy/val_best_epoch_{m}',
                    value=self.val_max_acc_dict[k][m],
                )

    def on_test_epoch_end(self) -> None:
        for test_index, regime_name in enumerate(self.regime_names):
            self.process_epoch_end(
                step_type='test', index=test_index, regime_name=regime_name
            )

    def log_roc(
        self, roc: cls_metrics.MulticlassROC | cls_metrics.BinaryROC, title: str
    ) -> None:
        ax_: plt.axes.Axes
        fig_: Any
        fig_, ax_ = roc.plot(score=True)
        ax_.set_title(title)
        ax_.set_xlabel('False Positive Rate')
        ax_.set_ylabel('True Positive Rate')

        # Add a straight line from (0,0) to (1,1) with the legend "Random"
        line = matplotlib_lines.Line2D(
            [0, 1], [0, 1], color='red', linestyle='--', label='Random'
        )
        ax_.add_line(line)
        ax_.legend()

        self.logger.experiment.log(  # type: ignore
            {
                title: wandb.Image(fig_, caption=title),
            }
        )
        # close the figure to prevent memory leaks
        plt.close(fig_)

    def format_confusion_matrix(self, cm: torch.Tensor) -> list[list]:
        """
        Formats a confusion matrix into a list of lists containing
        class names and their corresponding values.

        Args:
            cm (torch.Tensor): The confusion matrix to format.

        Returns:
            list[list]: A list of lists containing class names and their corresponding values.
        """
        class_names = self.class_names
        cm_list = cm.tolist()
        return [
            [class_names[i], class_names[j], cm_list[i][j]]
            for i, j in itertools.product(
                range(self.num_classes), range(self.num_classes)
            )
        ]

    def log_confusion_matrix(self, cm_data: list[list], title: str) -> None:
        """
        Logs a confusion matrix to the experiment logger.

        Args:
            cm_data (list[list]): A list of lists representing the confusion matrix.
                            Each inner list should contain the actual class name,
                            the predicted and the number of values.
            title (str): The title to use for the confusion matrix plot.

        Returns:
            None
        """
        if isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger.experiment
            fields = {
                'Actual': 'Actual',
                'Predicted': 'Predicted',
                'nPredictions': 'nPredictions',
            }

            wandb_logger.log(
                {
                    title: wandb.plot_table(
                        'EyeRead/multi-run-confusion-matrix',
                        wandb.Table(
                            columns=['Actual', 'Predicted', 'nPredictions'],
                            data=cm_data,
                        ),
                        fields,
                        {'title': title},
                    ),
                }
            )
        else:
            warnings.warn('No wandb logger found, cannot log confusion matrix')

    def configure_metrics(self):
        """Configures the metrics for the model."""
        if self.task == TaskTypes.BINARY_CLASSIFICATION:
            logger.info('Using binary metrics')
            metrics = torchmetrics.MetricCollection(
                {
                    'AUROC': cls_metrics.BinaryAUROC(
                        validate_args=self.validate_metrics,
                        # thresholds=10,
                    ),
                    'F1Score': cls_metrics.BinaryF1Score(
                        validate_args=self.validate_metrics,
                    ),
                    'Precision': cls_metrics.BinaryPrecision(
                        validate_args=self.validate_metrics,
                    ),
                    'Recall': cls_metrics.BinaryRecall(
                        validate_args=self.validate_metrics,
                    ),
                    'Accuracy': cls_metrics.BinaryAccuracy(
                        validate_args=self.validate_metrics,
                    ),
                }
            )

            confusion_matrix = cls_metrics.BinaryConfusionMatrix(
                validate_args=self.validate_metrics
            )

            roc = cls_metrics.BinaryROC(
                validate_args=self.validate_metrics,
            )
            # * Currently separate because expects preds or probs for each class which is not the case in binary case.
            balanced_accuracy = cls_metrics.MulticlassAccuracy(
                num_classes=self.num_classes,
                average=self.average,
                validate_args=self.validate_metrics,
            )

        elif self.task == TaskTypes.REGRESSION:
            logger.info('Using regression metrics')
            metrics = torchmetrics.MetricCollection(
                {
                    'RMSE': reg_metrics.MeanSquaredError(squared=False),
                    'R2Score': reg_metrics.R2Score(),
                }
            )

            confusion_matrix = None
            roc = None
            balanced_accuracy = None
        else:
            raise ValueError(f'Unknown task: {self.task}')

        return metrics, confusion_matrix, roc, balanced_accuracy

    def get_metrics_map(self, step_type: str, index=0) -> dict[str, Any]:
        if self.task == TaskTypes.REGRESSION:
            metrics_map = {
                'train': {
                    'cm': None,
                    'roc': None,
                    'metrics': self.train_metrics,
                    'balanced_accuracy': None,
                },
                'val': {
                    'cm': None,
                    'roc': None,
                    'metrics': self.val_metrics_list[index],
                    'balanced_accuracy': None,
                },
                'test': {
                    'cm': None,
                    'roc': None,
                    'metrics': self.test_metrics_list[index],
                    'balanced_accuracy': None,
                },
            }
        else:
            metrics_map = {
                'train': {
                    'cm': self.train_cm,
                    'roc': self.train_roc,
                    'metrics': self.train_metrics,
                    'balanced_accuracy': self.train_balanced_accuracy,
                },
                'val': {
                    'cm': self.val_cm_list[index],
                    'roc': self.val_roc_list[index],
                    'metrics': self.val_metrics_list[index],
                    'balanced_accuracy': self.val_balanced_accuracy_list[index],
                },
                'test': {
                    'cm': self.test_cm_list[index],
                    'roc': self.test_roc_list[index],
                    'metrics': self.test_metrics_list[index],
                    'balanced_accuracy': self.test_balanced_accuracy_list[index],
                },
            }

        return metrics_map[step_type]


class BaseMLModel(SharedBaseModel):
    """
    Base class for all ML models.
    """

    def __init__(
        self, model_args: MLModelArgs, trainer_args: TrainerML, data_args: DataArgs
    ):
        # Initialize shared base attributes
        SharedBaseModel.__init__(self, model_args=model_args, data_args=data_args)

        # ML-specific attributes
        self.num_workers = trainer_args.num_workers

        self.regime_names: list[str] = [
            'new_item',
            'new_subject',
            'new_item_and_subject',
            'all',
        ]  # This order is defined in the data module!

        self._init_classifier(model_args)
        self.balanced_class_accuracies = {}
        self.stage_count = {}

        self.ia_features = model_args.ia_features

        #### features builder ###
        self.use_item_level_features: bool = (
            len(model_args.item_level_features_modes) > 0
        )

        self.batch_size = model_args.batch_size
        self.feature_builder_device = 'cpu'

        self.acc_y_true_val = []
        self.acc_y_pred_val = []

        self.trial_level_feature_names: Union[None, list[str]] = None

        self.pca_explained_variance_ratio_threshold = (
            model_args.pca_explained_variance_ratio_threshold
        )
        self.pca = None
        self.model_args = model_args
        if self.task != TaskTypes.REGRESSION:
            self.label_encoder = LabelEncoder()
        self.loss_func = (
            metrics.mean_squared_error
            if self.task == TaskTypes.REGRESSION
            else metrics.log_loss
        )

    def _init_classifier(self, model_args) -> None:
        """
        Initialize the classifier.
        """
        # make pipeline from model_params.sklearn_pipeline
        ## empty pipeline
        self.classifier = Pipeline([])
        for step_name, method_import_path in model_args.sklearn_pipeline:
            module_name, class_name = method_import_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            class_obj = getattr(module, class_name)
            self.classifier.steps.append((step_name, class_obj()))

        # set params
        logger.info('Setting model params')
        model_args.init_sklearn_pipeline_params()
        self.classifier.set_params(**model_args.sklearn_pipeline_params)

    def shared_fit(self, dm) -> list:
        """
        Fit the model to the data.
        """
        logger.info('Fitting model')
        train_dataloader = DataLoader(
            dm.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        try:
            self.trial_level_feature_names = dm.train_dataset.trial_level_feature_names
        except AttributeError:
            logger.warning(
                'No trial level feature names found in the training dataset.'
            )

        train_batches = self.unpack_data(train_dataloader)
        return train_batches

    def predict(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Predict the model on the data.
        """
        dev_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        dev_batches = self.unpack_data(dev_dataloader)
        preds_list, probs_list = self.model_specific_predict(dev_batches)
        y_true_list = []
        for dev_batch in tqdm(dev_batches, desc='Feature extraction (pred)'):
            y_true_list.append(dev_batch.labels)

        preds = torch.cat(preds_list, dim=0)

        y_true = torch.cat(y_true_list, dim=0)
        if self.task != TaskTypes.REGRESSION:
            probs = torch.cat(probs_list, dim=0)
        else:
            probs = None

        return preds, probs, y_true

    def evaluate(self, eval_dataset, stage: str, validation_map: str) -> None:
        """
        Evaluate the model on the data.
        """
        self.stage = stage
        self.validation_map = validation_map

        (
            self.preds,
            self.probs,
            self.y_true,
        ) = self.predict(eval_dataset)

        if stage == 'val':
            self.acc_y_true_val.append(self.y_true)
            self.acc_y_pred_val.append(self.preds)

        self._on_eval_end()

    def _on_eval_end(self) -> None:
        """
        Function to run at the end of evaluation.
        """
        assert self.preds is not None
        assert self.y_true is not None

        # convert tensors to numpy
        self.preds = self.preds.numpy()
        self.y_true = self.y_true.numpy()

        if self.task != TaskTypes.REGRESSION:
            assert self.probs is not None
            self.probs = self.probs.numpy()

        self._log_metrics()

        self.stage_count[self.validation_map] = len(self.y_true)
        if self.task != TaskTypes.REGRESSION:
            self.balanced_class_accuracies[self.validation_map] = (
                metrics.balanced_accuracy_score(
                    y_true=self.y_true,
                    y_pred=self.preds,
                )
            )

        (
            self.preds,
            self.probs,
            self.y_true,
        ) = (None, None, None)

    def _log_metrics(self) -> None:
        """
        Log the metrics of the evaluation.
        """

        if self.trial_level_feature_names is not None:
            wandb.log(
                {
                    'Feature_names': self.trial_level_feature_names,
                }
            )
        if self.task == TaskTypes.REGRESSION:
            return

        # Log Confusion Matrices
        wandb.log(
            {
                f'Confusion_Matrix/{self.stage}_{self.validation_map}': wandb.plot.confusion_matrix(
                    preds=self.preds,
                    y_true=self.y_true,
                    class_names=self.class_names,
                    title=f'Confusion_Matrix/{self.stage}_{self.validation_map}',
                )
            }
        )

        # Log ROC curve
        try:
            wandb.log(
                {
                    f'ROC_Curve/{self.stage}_{self.validation_map}': wandb.plot.roc_curve(
                        y_true=self.y_true,
                        y_probas=self.probs,
                        labels=self.class_names,
                        title=f'ROC_Curve/{self.stage}_{self.validation_map}',
                    )
                }
            )

        except TypeError:
            logger.warning('ROC curve not calculated!!')

        # per class metrics (Unordered)
        class_names_unordered = np.array(self.class_names)[np.unique(self.y_true)]
        self.per_class_metrics = metrics.classification_report(
            y_true=self.y_true,
            y_pred=self.preds,
            target_names=class_names_unordered,
            output_dict=True,
        )
        for class_name, metrics_dict in self.per_class_metrics.items():
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    wandb.summary[
                        f'{metric_name}_{class_name}/{self.stage}_{self.validation_map}'
                    ] = value
            else:
                wandb.summary[f'{class_name}/{self.stage}_{self.validation_map}'] = (
                    metrics_dict
                )

    def on_stage_end(self) -> None:
        """
        Function to run at the end of a stage.
        """
        if self.task != TaskTypes.REGRESSION:
            # Log balanced classless accuracy
            if self.stage == 'val':
                wandb.summary[f'Balanced_Accuracy/{self.stage}_average'] = np.mean(
                    list(self.balanced_class_accuracies.values())
                )
                wandb.summary[f'Balanced_Accuracy/{self.stage}_weighted_average'] = (
                    np.average(
                        list(self.balanced_class_accuracies.values()),
                        weights=list(self.stage_count.values()),
                    )
                )

            for eval_regime, metric_val in self.balanced_class_accuracies.items():
                wandb.summary[f'Balanced_Accuracy/{self.stage}_{eval_regime}'] = (
                    metric_val
                )

        if self.stage == 'val':
            all_val_y_true = torch.cat(self.acc_y_true_val)
            all_val_y_pred = torch.cat(self.acc_y_pred_val)

            if self.task != TaskTypes.REGRESSION:
                wandb.summary['Balanced_Accuracy/val_all'] = (
                    metrics.balanced_accuracy_score(
                        y_true=all_val_y_true, y_pred=all_val_y_pred
                    )
                )

            if self.model_args.use_class_weighted_loss:
                # Map each y_true to its class weight
                sample_weight = np.array(
                    [self.class_weights[int(label)] for label in all_val_y_true]
                )
                loss = self.loss_func(
                    y_true=all_val_y_true,
                    y_pred=all_val_y_pred,
                    sample_weight=sample_weight,
                )
            else:
                loss = self.loss_func(y_true=all_val_y_true, y_pred=all_val_y_pred)
            wandb.summary['loss/val_all'] = loss

        # empty caches
        self.balanced_class_accuracies = {}
        self.stage_count = {}

    def unpack_data(self, dataloader) -> list:
        """
        Unpacks the batch into the different tensors.
        """
        return [BaseModel.unpack_batch(batch) for batch in dataloader]

    def _features_builder(self, train_batch) -> torch.Tensor:
        """
        Concatenate features for the model from different sources.
        """
        if train_batch.labels.ndim == 0:
            for key, value in train_batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    train_batch.__dict__[key] = value.unsqueeze(0)

        features_list = []
        if self.use_item_level_features:
            if hasattr(train_batch, 'trial_level_features'):
                features_list.append(
                    train_batch.trial_level_features.squeeze().to(
                        self.feature_builder_device
                    )
                )
            else:
                raise ValueError('No trial level features found in the batch data.')

        assert len(features_list) > 0, 'No features found for the model.'
        features = torch.cat(features_list)

        if features.ndim == 1:
            features = features.unsqueeze(1)

        return features

    def _prepare_features_and_labels(
        self, batches, training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Prepare features and labels from a list of batches.
        """
        features_list: list[torch.Tensor] = []
        y_true_list: list[torch.Tensor] = []
        trial_groups_keys_list: list[np.ndarray] = []
        iterator = tqdm(batches, desc='Feature extraction') if training else batches
        for batch in iterator:
            features = self._features_builder(batch).to('cpu')

            trial_groups_keys_list.append(
                np.transpose(np.array(batch.batch_item_keys), axes=(1, 0))
            )

            features_list.append(features)
            y_true_list.append(batch.labels)
        features = torch.cat(features_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        trial_groups_keys = np.concatenate(trial_groups_keys_list)

        trial_key_columns: list[str] = list(
            np.array(batches[0].trial_groupby_columns)[:, 0]
        )

        return (
            features.numpy(),
            y_true.numpy(),
            trial_groups_keys,
            trial_key_columns,
        )

    def _predict_with_fallback(
        self, features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Try to use predict_proba, if not available use predict and convert to probabilities.
        """
        try:
            probs = torch.tensor(self.classifier.predict_proba(features))
            preds = probs.argmax(dim=1)
        except AttributeError:
            # predict_proba not available
            preds = torch.tensor(self.classifier.predict(features), dtype=torch.int64)
            probs = torch.zeros((len(preds), self.num_classes))
            probs[range(len(preds)), preds.int()] = 1
        return preds, probs

    def fit(self, dm) -> None:
        """
        Default shared_fit implementation.
        Most subclasses can use this unless they have special requirements.
        """
        train_batches = self.shared_fit(dm)
        features, y_true, trial_groups_keys, trial_key_columns = (
            self._prepare_features_and_labels(train_batches, training=True)
        )
        if self.task != TaskTypes.REGRESSION:
            y_true = self.label_encoder.fit_transform(y_true)

        if self.pca_explained_variance_ratio_threshold < 1.0:
            features = self._apply_pca(features)

        if self.model_args.use_class_weighted_loss:
            classes_weights = class_weight.compute_sample_weight('balanced', y=y_true)
            self.classifier.fit(features, y_true, clf__sample_weight=classes_weights)
        else:
            self.classifier.fit(features, y_true)

    def model_specific_predict(
        self, dev_batches: list
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        features_list = []
        for batch in dev_batches:
            features = self._features_builder(batch).to('cpu').numpy()

            if self.pca_explained_variance_ratio_threshold < 1.0:
                features = self._apply_pca(features)
            features_list.append(features)

        for features in features_list:
            if self.task == TaskTypes.REGRESSION:
                preds = torch.tensor(self.classifier.predict(features))
                probs = None
            else:
                preds, probs = self._predict_with_fallback(features)
                preds = torch.tensor(self.label_encoder.inverse_transform(preds))

            probs_list.append(probs)
            preds_list.append(preds)

        return preds_list, probs_list

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        if self.pca is None:
            self.pca = PCA(n_components=self.pca_explained_variance_ratio_threshold)
            features = self.pca.fit_transform(features)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            # check if wandb is available
            if wandb.run is not None:
                wandb.log({'PCA_explained_variance': explained_variance})
                wandb.log({'PCA_num_components': self.pca.n_components_})
        else:
            features = self.pca.transform(features)
        return features


class ModelFactory:
    """A factory class to register and retrieve models."""

    models = {}

    @classmethod
    def add(cls, model: Type[BaseModel | BaseMLModel]) -> None:
        """Register a model class."""
        cls.models[model.__name__] = model

    @classmethod
    def get(cls, model_name: str) -> Type[BaseModel | BaseMLModel]:
        """Retrieve a model class by its name."""
        model = cls.models[model_name]
        return model


def register_model(
    model: Type[BaseModel | BaseMLModel],
) -> Type[BaseModel | BaseMLModel]:
    """Decorator to register a model class."""
    ModelFactory.add(model)
    return model
