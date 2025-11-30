"""Ahn et al. baseline models
Based on
https://github.com/aeye-lab/etra-reading-comprehension/blob/main/ahn_baseline/evaluate_ahn_baseline.py
https://github.com/ahnchive/SB-SAT/blob/master/model/model_training.ipynb
"""

import torch
from torch import nn

from src.configs.data import DataArgs
from src.configs.models.dl.Ahn import Ahn, AhnCNN, AhnRNN
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


class AhnModel(BaseModel):
    """
    Base model for Ahn et al.

    Args:
        model_args (AhnArgs): The model arguments.
        trainer_args (TrainerDL): The trainer arguments.
        data_args (DataArgs): The data arguments.
    """

    def __init__(
        self,
        model_args: Ahn,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
        self.model_args = model_args
        self.input_dim = (
            model_args.fixation_dim
            if model_args.use_fixation_report
            else model_args.eyes_dim
        )
        self.preorder = model_args.preorder
        self.model: nn.Module

        self.train()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared step for training and validation.

        Args:
            batch (list): The input batch.

        Returns:
            tuple: A tuple containing ordered labels, loss, ordered logits, labels, and logits.
        """
        batch_data = self.unpack_batch(batch)
        assert batch_data.fixation_features is not None, 'eyes_tensor not in batch_dict'
        labels = batch_data.labels
        logits, unused_hidden = self(x=batch_data.fixation_features)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, labels)

        return labels, loss, logits


@register_model
class AhnRNNModel(AhnModel):
    """
    RNN model for Ahn et al. baseline

    Args:
        model_args (AhnRNN): The model arguments.
        trainer_args (TrainerDL): The trainer arguments.
        data_args (DataArgs): The data arguments.
    """

    def __init__(
        self,
        model_args: AhnRNN,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(model_args, trainer_args, data_args=data_args)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.model_args.hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=model_args.num_lstm_layers,
        )
        self.fc = nn.Sequential(
            nn.Dropout(self.model_args.fc_dropout),  # (batch_size, hidden_size * 2)
            nn.Linear(
                model_args.hidden_dim * 2, model_args.hidden_dim * 2
            ),  # (batch_size, 50)
            nn.ReLU(),
            nn.Dropout(self.model_args.fc_dropout),
            nn.Linear(
                model_args.hidden_dim * 2,
                model_args.fc_hidden_dim,
            ),  # (batch_size, 2)
            nn.ReLU(),
            nn.Linear(model_args.fc_hidden_dim, self.num_classes),  # (batch_size, 2)
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the output tensor and hidden representations.
        """
        # take the last hidden state of the lstm
        x, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        x = x[:, -1, :]  # (batch_size, hidden_size * 2)
        hidden_representations = x.clone()
        x = self.fc(x)  # (batch_size, 2)
        return x, hidden_representations


@register_model
class AhnCNNModel(AhnModel):
    """
    CNN model for Ahn et al. baseline

    Args:
        model_args (AhnCNN): The model arguments.
        trainer_args (TrainerDL): The trainer arguments.
        data_args (DataArgs): The data arguments.
    """

    def __init__(
        self,
        model_args: AhnCNN,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(model_args, trainer_args, data_args=data_args)

        self.input_dim = self.input_dim
        self.model_args = model_args
        hidden_dim = self.model_args.hidden_dim
        kernel_size = self.model_args.conv_kernel_size
        fc_dropout = self.model_args.fc_dropout
        fc_hidden_dim1 = self.model_args.fc_hidden_dim1
        fc_hidden_dim2 = self.model_args.fc_hidden_dim2

        self.conv_model = nn.Sequential(
            # (batch size, number of features, max seq len)
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
            ),  # (batch size, hidden_dim, max seq len - 2)
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size
            ),  # (batch size, hidden_dim, max seq len - 4)
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size
            ),  # (batch size, hidden_dim, max seq len - 6)
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=self.model_args.pooling_kernel_size
            ),  # (batch size, hidden_dim, (max seq len -6) / 2)
            nn.Dropout(fc_dropout),  # (batch size, hidden_dim, (max seq len -6) / 2)
            nn.Flatten(),  # (batch size, hidden_dim * ((max seq len -6) / 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(
                ((self.max_scanpath_length - 6) // 2) * hidden_dim, fc_hidden_dim1
            ),  # (batch size, 50)
            nn.ReLU(),
            nn.Dropout(fc_dropout),  # (batch size, 50)
            nn.Linear(fc_hidden_dim1, fc_hidden_dim2),  # (batch size, 20)
            nn.ReLU(),
            nn.Linear(fc_hidden_dim2, self.num_classes),  # (batch size, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the output tensor and hidden representations.
        """
        x = x.transpose(1, 2)  # (batch size, number of features, max seq len)
        hidden_representations = self.conv_model(x)
        x = self.fc(hidden_representations)
        return x, hidden_representations
