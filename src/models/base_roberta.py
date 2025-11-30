"""base_roberta.py - Base class for MAG and RoBERTeye models.
See 1. On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines:
https://www.semanticscholar.org/reader/8b9d77d5e52a70af37451d3db3d32781b83ea054 for parameters
"""

import torch
from transformers.optimization import get_linear_schedule_with_warmup

from src.configs.constants import DLModelNames, TaskTypes
from src.configs.data import DataArgs
from src.configs.models.dl.MAG import MAG
from src.configs.models.dl.PostFusion import PostFusion
from src.configs.models.dl.RoBERTeye import RoberteyeArgs
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel


class BaseMultiModalRoberta(BaseModel):
    """
    Model for Multiple Choice Question Answering and question prediction tasks.

    Args:
        model_args (Roberteye | MAG | PostFusion): Model arguments.
        trainer_args (TrainerDL): Trainer arguments.
    """

    def __init__(
        self,
        model_args: RoberteyeArgs | MAG | PostFusion,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

        self.model_args = model_args
        self.preorder = model_args.preorder
        self.warmup_proportion = model_args.warmup_proportion

        self.train()
        self.save_hyperparameters()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        gaze_features,
        gaze_positions,
        eye_token_type_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels.
            gaze_features (torch.Tensor): Gaze features.
            gaze_positions (torch.Tensor): Gaze positions.
            eye_token_type_ids (torch.Tensor, optional): Eye token type IDs. Defaults to None.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            eye_token_type_ids=eye_token_type_ids,
            **kwargs,
        )

    def finalize_eye_data(
        self, batch
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """
        Set the final gaze_features, gaze_positions, and attention_mask
        based on whether we use fixation reports or not.

        Args:
            batch (BatchData): Batch data.
            actual_needed_eye_padding_len (int | None): Actual needed eye padding length.

        Returns:
            tuple: Gaze features, gaze positions, and attention mask.
        """
        if self.model_args.use_fixation_report:
            assert batch.fixation_features is not None, (
                'fixation_features must be present if using fixation_report'
            )

            gaze_positions = batch.grouped_inversions
            gaze_features = batch.fixation_features

            attention_mask = batch.input_masks

        else:
            # No fixation report => we must have eyes instead
            assert batch.eyes is not None
            assert batch.input_ids is not None

            gaze_features = batch.eyes
            gaze_positions = batch.grouped_inversions
            attention_mask = batch.input_masks

        # If we do NOT want to prepend_eye_features_to_text for ROBERTEYE_MODEL, set them to None
        if (
            not self.model_args.prepend_eye_features_to_text
            and self.model_args.base_model_name == DLModelNames.ROBERTEYE_MODEL
        ):
            gaze_features = None
            gaze_positions = None

        return gaze_features, gaze_positions, attention_mask

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared step for training, validation, and testing.

        Args:
            batch (list): Batch data.

        Returns:
            tuple: Ordered labels, loss, ordered logits, labels, and logits.
        """
        # 1. Unpack the batch
        batch_data = self.unpack_batch(batch)

        # 2. Process different modes that affect input_masks/grouped_inversions
        eye_token_type_ids = (
            None  # if it will stay None, it will be created automatically later
        )

        # Safety checks
        assert batch_data.input_masks is not None, 'input_masks cannot be None'
        assert batch_data.grouped_inversions is not None, (
            'grouped_inversions cannot be None'
        )

        # 3. Finalize eye/gaze data
        gaze_features, gaze_positions, attention_mask = self.finalize_eye_data(
            batch_data
        )

        labels = batch_data.labels

        if self.task == TaskTypes.REGRESSION:
            labels = labels.squeeze().float()

        output = self(
            input_ids=batch_data.input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            output_hidden_states=True,
            eye_token_type_ids=eye_token_type_ids,
        )

        logits = output.logits
        if self.task == TaskTypes.REGRESSION:
            logits = logits.squeeze()

        loss = self.loss(logits, labels)

        return labels, loss, logits

    def configure_optimizers(self) -> tuple[list, list]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        # Define the optimizer
        assert self.warmup_proportion is not None

        # Copied from bert
        param_optimizer = list(self.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.1,
            },
            {
                'params': [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(stepping_batches * self.warmup_proportion),
            num_training_steps=stepping_batches,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
