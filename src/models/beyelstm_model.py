"""Beye LSTM baseline model.
Based on https://github.com/aeye-lab/etra-reading-comprehension/blob/main/nn/model.py
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.configs.data import DataArgs
from src.configs.models.dl.BEyeLSTM import (
    BEyeLSTMArgs,
)
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


@register_model
class BEyeLSTMModel(BaseModel):
    """Beye model."""

    def __init__(
        self,
        model_args: BEyeLSTMArgs,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
        self.preorder = False
        self.model_args = model_args
        self.pos_block = LSTMBlock(model_args, num_embed=model_args.num_pos)
        self.content_block = LSTMBlock(model_args, num_embed=model_args.num_content)
        self.fixations_block = LSTMBlock(model_args, input_dim=model_args.fixations_dim)

        self.gsf_block = nn.Sequential(
            nn.Dropout(p=model_args.dropout_rate),
            nn.Linear(
                in_features=model_args.gsf_dim, out_features=model_args.gsf_out_dim
            ),
            nn.ReLU(),
        )
        fc1_in_features = model_args.lstm_block_fc2_out_dim * 3 + model_args.gsf_out_dim
        self.fc1 = nn.Linear(
            in_features=fc1_in_features,
            out_features=model_args.after_cat_fc_hidden_dim,
        )
        self.fc2 = nn.Linear(
            in_features=model_args.after_cat_fc_hidden_dim,
            out_features=self.num_classes,
        )

        print(f'##### Preorder labels: {self.preorder} #####')

        self.train()
        self.save_hyperparameters()

    def forward(  # type: ignore
        self,
        x_pos: torch.Tensor,
        x_content: torch.Tensor,
        x_gsf: torch.Tensor,
        x_fixations: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for NNModel.

        Args:
            x_pos (torch.Tensor): Position tensor.
            x_content (torch.Tensor): Content tensor.
            x_gsf (torch.Tensor): GSF tensor.
            x_fixations (torch.Tensor): Fixations tensor (batch size x MAX_SCANPATH_LEN x 4).
                                        Padded with 0s
            seq_lengths (torch.Tensor): Length of scanpath for each trial.

        Returns:
            torch.Tensor: Output tensor.
        """
        concat_list = []
        concat_list.append(self.pos_block(x_pos, seq_lengths=seq_lengths))
        concat_list.append(self.content_block(x_content, seq_lengths=seq_lengths))
        concat_list.append(self.gsf_block(x_gsf.squeeze(1)))
        concat_list.append(self.fixations_block(x_fixations, seq_lengths=seq_lengths))
        trial_embd = torch.cat(concat_list, dim=1)
        x = F.relu(self.fc1(trial_embd))
        x = self.fc2(x)
        return x, trial_embd

    def shared_step(
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        assert batch_data.fixation_features is not None, 'eyes_tensor not in batch_dict'
        assert batch_data.scanpath_pads is not None, 'scanpath_pads not in batch_dict'
        labels = batch_data.labels

        shortest_scanpath_pad = batch_data.scanpath_pads.min()
        longest_batch_scanpath: int = int(
            self.max_scanpath_length - shortest_scanpath_pad
        )

        fixation_features = batch_data.fixation_features[
            ..., :longest_batch_scanpath, :
        ]
        scanpath_lengths = (
            batch_data.fixation_features.shape[1] - batch_data.scanpath_pads
        )
        logits, trial_embd = self(
            x_fixations=fixation_features[..., :4],
            x_content=fixation_features[..., -2].int(),
            x_pos=fixation_features[..., -1].int(),
            x_gsf=batch_data.trial_level_features,
            seq_lengths=scanpath_lengths,
        )

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, labels)

        return labels, loss, logits.squeeze()


class LSTMBlock(nn.Module):
    """LSTM block for the Beye model."""

    def __init__(
        self,
        model_args: BEyeLSTMArgs,
        input_dim: int | None = None,
        num_embed: int | None = None,
    ) -> None:
        """Initialize LSTMBlock.

        Args:
            model_args (BEyeLSTMArgs): Model parameters.
            input_dim (int | None, optional): Input dimension. Defaults to None.
            num_embed (int | None, optional): Embedding dimension. Defaults to None.
        """
        super().__init__()
        assert (input_dim is None) != (num_embed is None), (
            'input_dim and num_embeddings cannot both be None or not None.'
        )
        self.num_embeddings = num_embed  # for universal_pos and Content
        if num_embed:
            self.embedding = nn.Embedding(num_embed, model_args.embedding_dim)
            lstm_input_dim = model_args.embedding_dim
        else:  # for Fixations
            lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            model_args.hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(model_args.dropout_rate)
        self.fc1 = nn.Linear(
            2 * model_args.hidden_dim, model_args.lstm_block_fc1_out_dim
        )
        self.fc2 = nn.Linear(
            model_args.lstm_block_fc1_out_dim, model_args.lstm_block_fc2_out_dim
        )
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor, seq_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for LSTMBlock.

        Args:
            seq_lengths (torch.Tensor | None): Length of scanpath for each trial. Defaults to None.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.num_embeddings:
            x = self.embedding(x)

        if seq_lengths is not None:
            sorted_lengths, indices = torch.sort(seq_lengths, descending=True)
            x = x[indices]
            # Pass the entire sequence through the LSTM layer
            packed_x = nn.utils.rnn.pack_padded_sequence(
                input=x,
                lengths=sorted_lengths.to('cpu'),
                batch_first=True,
                enforce_sorted=True,
            )
            assert not torch.isnan(packed_x.data).any()

            unused_packed_output, (ht, unused_ct) = self.lstm(packed_x)

            # from dimension (2, batch_size, hidden_dim) to (batch_size, 2*hidden_dim)
            x = torch.cat((ht[0], ht[1]), dim=1)
            x = x[torch.argsort(indices)]
        else:
            unused_output, (h, unused_c) = self.lstm(x)
            h_concat = torch.cat((h[0], h[1]), dim=1)
            x = h_concat

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x
