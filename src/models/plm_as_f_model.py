from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.configs.constants import (
    BINARY_P_AND_Q_TASKS,
    BINARY_PARAGRAPH_ONLY_TASKS,
    REGRESSION_PARAGRAPH_ONLY_TASKS,
    SCANPATH_PADDING_VAL,
    PredMode,
)
from src.configs.data import DataArgs
from src.configs.models.dl.PLMASF import PLMASfArgs
from src.configs.trainers import TrainerDL
from src.models import plm_as_model
from src.models.base_model import BaseModel, register_model


@register_model
class PLMASFModel(BaseModel):
    def __init__(
        self,
        model_args: PLMASfArgs,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

        self.model_args = model_args
        self.backbone = model_args.backbone
        self.freeze_bert = model_args.freeze

        self.add_question = data_args.task == PredMode.RC

        self.fast_tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.pad_token_id = self.fast_tokenizer.pad_token_id
        self.sep_token_id = self.fast_tokenizer.sep_token_id
        # Cache tokenizer with add_prefix_space for scanpath processing
        self.fast_tokenizer_prefix = AutoTokenizer.from_pretrained(
            self.backbone, add_prefix_space=True
        )

        self.classifier_head = nn.Linear(
            self.model_args.lstm_hidden_size * 2, self.num_classes
        )  # *2 for bidirectional
        self.bert_dim = model_args.text_dim

        encoder_config = AutoConfig.from_pretrained(self.backbone)
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print('keeping Bert with pre-trained weights')
        self.bert_encoder = AutoModel.from_pretrained(
            self.backbone, config=encoder_config
        )  # type: ignore

        # freeze the parameters in Bert model
        if self.freeze_bert:
            for param in self.bert_encoder.parameters():  # type: ignore
                param.requires_grad = False

        # Create feature index mappings for safe access
        self.fixation_feature_indices = {
            name: idx for idx, name in enumerate(self.model_args.fixation_features)
        }
        self.eye_feature_indices = {
            name: idx for idx, name in enumerate(self.model_args.eye_features)
        }

        # Calculate number of additional features dynamically
        # Based on organise_sp_fixation_features:
        # features_by_sp_idx: 8 features (from fixation_features)
        # features_by_word_idx: 2 features (from eye_features)
        self.num_features_by_sp_idx = 8
        self.num_features_by_word_idx = 2
        self.num_additional_fixations_features = (
            self.num_features_by_sp_idx + self.num_features_by_word_idx
        )

        # project bert_dim to bert_dim+num_additional_fixations_features
        self.projection_layer = nn.Sequential(
            nn.Linear(
                self.bert_dim, self.bert_dim + self.num_additional_fixations_features
            )
        )
        # create fse_lstm
        self.fse_lstm = nn.LSTM(
            input_size=self.bert_dim + self.num_additional_fixations_features,
            hidden_size=self.model_args.lstm_hidden_size,
            num_layers=self.model_args.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.model_args.lstm_dropout,
        )

        self.train()
        self.save_hyperparameters()

    def organise_sp_fixation_features(
        self, fixation_features: torch.Tensor, ia_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Organize scanpath and fixation features for the model.

        Args:
            fixation_features: Tensor of fixation features indexed by scanpath position
            ia_features: Tensor of interest area features indexed by word position

        Returns:
            features_by_sp_idx: Features indexed by scanpath position (8 features)
            features_by_word_idx: Features indexed by word position (2 features)
        """

        # Create safe feature getters
        def get_fix_feat(name: str) -> torch.Tensor:
            if name not in self.fixation_feature_indices:
                raise ValueError(
                    f"Feature '{name}' not found in fixation_features. "
                    f'Available: {list(self.fixation_feature_indices.keys())}'
                )
            return fixation_features[..., self.fixation_feature_indices[name]]

        def get_eye_feat(name: str) -> torch.Tensor:
            if name not in self.eye_feature_indices:
                raise ValueError(
                    f"Feature '{name}' not found in eye_features. "
                    f'Available: {list(self.eye_feature_indices.keys())}'
                )
            return ia_features[..., self.eye_feature_indices[name]]

        features_by_sp_idx = []
        features_by_word_idx = []

        # Following the PLMAS-F paper:
        # 1. Horizontal position of the fixation (CURRENT_FIX_Y) - fixation_features
        features_by_sp_idx.append(get_fix_feat('CURRENT_FIX_Y'))

        # 2. Total gaze duration (sum of all fixations on the word) (IA_DWELL_TIME) - ia_features
        features_by_word_idx.append(get_eye_feat('IA_DWELL_TIME'))

        # 3. Landing position of first fixation within the word (IA_FIRST_RUN_LANDING_POSITION) - ia_features
        # features_by_word_idx.append(ia_features[..., 1])  #! removed because of nans

        # 4. Landing position of last fixation within the word (IA_LAST_RUN_LANDING_POSITION) - ia_features
        # features_by_word_idx.append(ia_features[..., 2])  #! removed because of nans

        # 5. Duration of first fixation (IA_FIRST_FIXATION_DURATION) - ia_features
        features_by_word_idx.append(get_eye_feat('IA_FIRST_FIXATION_DURATION'))

        # 6. Duration of outgoing saccade (NEXT_SAC_DURATION) - fixation_features
        features_by_sp_idx.append(get_fix_feat('NEXT_SAC_DURATION'))

        # 7. Horizontal distance of outgoing saccade (NEXT_SAC_END_X - NEXT_SAC_START_X) - fixation_features
        features_by_sp_idx.append(
            get_fix_feat('NEXT_SAC_END_X') - get_fix_feat('NEXT_SAC_START_X')
        )

        # 8. Vertical distance of outgoing saccade (NEXT_SAC_END_Y - NEXT_SAC_START_Y) - fixation_features
        features_by_sp_idx.append(
            get_fix_feat('NEXT_SAC_END_Y') - get_fix_feat('NEXT_SAC_START_Y')
        )

        # 9. Total distance of outgoing saccade (pitagoras of 7 and 8) - fixation_features
        features_by_sp_idx.append(
            torch.sqrt(
                torch.pow(
                    get_fix_feat('NEXT_SAC_END_X') - get_fix_feat('NEXT_SAC_START_X'), 2
                )
                + torch.pow(
                    get_fix_feat('NEXT_SAC_END_Y') - get_fix_feat('NEXT_SAC_START_Y'), 2
                )
            )
        )

        # 10. Duration of incoming saccade (NEXT_SAC_DURATION of previous fixation) - fixation_features
        ## v[i] = v[i-1], v[0] = 0
        tmp = torch.zeros_like(get_fix_feat('NEXT_SAC_DURATION')).to(
            fixation_features.device
        )
        tmp[..., 1:] = get_fix_feat('NEXT_SAC_DURATION')[..., :-1]
        features_by_sp_idx.append(tmp)

        # 11. Horizontal distance of incoming saccade (NEXT_SAC_END_X - NEXT_SAC_START_X of previous) - fixation_features
        prev_sac_end_x = torch.zeros_like(get_fix_feat('NEXT_SAC_END_X')).to(
            fixation_features.device
        )
        prev_sac_end_x[..., 1:] = get_fix_feat('NEXT_SAC_END_X')[..., :-1]
        prev_sac_start_x = torch.zeros_like(get_fix_feat('NEXT_SAC_START_X')).to(
            fixation_features.device
        )
        prev_sac_start_x[..., 1:] = get_fix_feat('NEXT_SAC_START_X')[..., :-1]
        features_by_sp_idx.append(prev_sac_end_x - prev_sac_start_x)

        # 12. Vertical distance of incoming saccade (NEXT_SAC_END_Y - NEXT_SAC_START_Y of previous) - fixation_features
        prev_sac_end_y = torch.zeros_like(get_fix_feat('NEXT_SAC_END_Y')).to(
            fixation_features.device
        )
        prev_sac_end_y[..., 1:] = get_fix_feat('NEXT_SAC_END_Y')[..., :-1]
        prev_sac_start_y = torch.zeros_like(get_fix_feat('NEXT_SAC_START_Y')).to(
            fixation_features.device
        )
        prev_sac_start_y[..., 1:] = get_fix_feat('NEXT_SAC_START_Y')[..., :-1]
        features_by_sp_idx.append(prev_sac_end_y - prev_sac_start_y)

        features_by_sp_idx_tensor = torch.stack(features_by_sp_idx, dim=2)
        features_by_word_idx_tensor = torch.stack(features_by_word_idx, dim=2)

        return features_by_sp_idx_tensor, features_by_word_idx_tensor

    def split_context_embeds(
        self,
        encoded_word_seq: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[
        Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]
    ]:
        # Find the positions of the separator tokens
        sep_positions = (
            (input_ids == self.sep_token_id).nonzero(as_tuple=True)[0].cpu().numpy()
        )

        # Calculate the sizes of the splits
        split_sizes = np.diff(a=sep_positions, prepend=0, append=input_ids.size(dim=0))
        assert split_sizes.sum() == input_ids.size(dim=0), (
            f'split_sizes.sum(): {split_sizes.sum()}'
        )
        if self.add_question:
            assert split_sizes.size == 4, (
                f'split_sizes.size: {split_sizes.size} but expected 4'
            )
            # Split the encoded_word_seq tensor at the separator positions
            p, _, q, _ = torch.split(
                tensor=encoded_word_seq,
                split_size_or_sections=split_sizes.tolist(),
                dim=0,
            )
            a = torch.zeros_like(q)

        else:
            # only paragraph
            assert split_sizes.size == 4, (
                f'split_sizes.size: {split_sizes.size} but expected 4'
            )
            # Split the encoded_word_seq tensor at the separator positions
            p, _, _, _ = torch.split(
                tensor=encoded_word_seq,
                split_size_or_sections=split_sizes.tolist(),
                dim=0,
            )
            # q = torch.zeros_like(p)
            # q = q[1:, :].mean(dim=0)
            # a = torch.zeros_like(p)
            q = None
            a = None

        return p, q, a

    def split_context_embds_batched(self, encoded_word_seq, input_ids):
        p_embds, q_embds, a_embeds, p_masks, q_masks, a_masks = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # initialize so that the return variables are defined
        p_embds_batches, q_embds_batches, a_embeds_batches = [], [], []
        # Process each batch separately
        for ewsb, iib in zip(encoded_word_seq, input_ids):
            p_embds, q_embds, a_embeds = self.split_context_embeds(
                encoded_word_seq=ewsb, input_ids=iib
            )

            p_embds_batches.append(p_embds)
            q_embds_batches.append(q_embds)
            a_embeds_batches.append(a_embeds)

        # pad the embeddings to the maximum sequence length of each list
        p_max_len = self.actual_max_needed_len
        p_masks = torch.stack(
            [
                torch.cat([torch.ones(p.shape[0]), torch.zeros(p_max_len - p.shape[0])])
                for p in p_embds_batches
            ],
            dim=0,
        )
        p_embds_batches = [
            F.pad(p, (0, 0, 0, p_max_len - p.shape[0])) for p in p_embds_batches
        ]
        p_embds = torch.stack(p_embds_batches, dim=0)

        if self.add_question:
            q_max_len = max([q.shape[0] for q in q_embds_batches])
            q_masks = torch.stack(
                [
                    torch.cat(
                        [torch.ones(q.shape[0]), torch.zeros(q_max_len - q.shape[0])]
                    )
                    for q in q_embds_batches
                ],
                dim=0,
            )
            q_embds_batches = [
                F.pad(q, (0, 0, 0, q_max_len - q.shape[0])) for q in q_embds_batches
            ]
            q_embds = torch.stack(q_embds_batches, dim=0)

        return p_embds, q_embds, q_embds, p_masks, q_masks, a_masks

    def forward(
        self,
        input_ids,
        input_masks,
        p_input_ids,
        scanpath,
        fixation_features,
        scanpath_pads,
        eyes,
    ):
        assert (
            input_ids[:, 0].sum().item() == 0
        )  # The CLS token is always present first (and 0 in roberta)

        scanpath, fixation_features = trim_scanpath_and_fixation_features(
            scanpath=scanpath,
            fixation_features=fixation_features,
            scanpath_pads=scanpath_pads,
            max_scanpath_length=self.max_scanpath_length,
        )

        features_by_sp_idx, features_by_word_idx = self.organise_sp_fixation_features(
            fixation_features=fixation_features,
            ia_features=eyes,
        )

        # scanpath masks --------------------------
        scanpath_masks = torch.ones_like(scanpath)
        scanpath_masks[scanpath == self.pad_token_id] = 0
        # -----------------------------------------

        # get the decoded texts from the input_ids
        p_input_ids_decoded_txts = self.fast_tokenizer.batch_decode(
            p_input_ids, return_tensors='pt'
        )  # previously decoded_to_txt_input_ids
        # -----------------------------------------

        paragraph_token_to_word_idxs_in_p_text = (
            plm_as_model.align_word_ids_with_input_ids(
                tokenizer=self.fast_tokenizer,
                input_ids=p_input_ids,
                decoded_to_txt_input_ids=p_input_ids_decoded_txts,
            )
        )  # previously word_ids_sn

        # in the decoded texts, space between <pad><pad>, <pad><s>, etc.
        p_input_ids_decoded_txts = list(
            map(
                lambda x: x.replace('<', ' <').split(' ')[
                    1:
                ],  # <s> is the first token so, when doing the replace, we get " <s>", so we split by space and take the second element
                p_input_ids_decoded_txts,
            )
        )

        scanpath_token_to_word_idxs_in_sp_text, scanpath_input_ids = (
            plm_as_model.calc_sp_word_input_ids(
                input_ids=p_input_ids,
                decoded_to_txt_input_ids=p_input_ids_decoded_txts,
                roberta_tokenizer_prefix_space=self.fast_tokenizer_prefix,
                scanpath=scanpath,
            )
        )  # previously word_ids_sp, sp_input_ids

        # 1. encode the whole sequence (input_ids)
        # 2. then split the embeddings into paragraph, question, and answer embeddings
        # 3. reorder the paragraph embeddings according to the scanpath
        # 4. combine the paragraph embeddings with features_by_sp_idx, features_by_word_idx
        # 5. combine parts into a sequence of embeddings to feed the lstm

        # (1) encode the whole sequence
        with torch.no_grad():
            outputs = self.bert_encoder(input_ids=input_ids, attention_mask=input_masks)
            #  Make the embedding of the <pad> token to be zeros
        outputs.last_hidden_state[input_ids == self.pad_token_id] = 0
        encoded_input_ids = outputs.last_hidden_state

        # (2) split the embeddings into paragraph, question, and answer embeddings
        p_embds, q_embds, a_embeds, p_masks, q_masks, a_masks = (
            self.split_context_embds_batched(encoded_input_ids, input_ids)
        )  # embds shape: [batch, seq_len, emb_dim]

        # (3) reorder the paragraph embeddings according to the scanpath
        # Pool bert subword to word level for english corpus
        merged_p_emb, p_mask = plm_as_model.pool_subword_to_word(
            p_embds,
            paragraph_token_to_word_idxs_in_p_text,
            target='sn',
            max_seq_len=self.actual_max_needed_len,
            bert_dim=self.bert_dim,
            pool_method='sum',
        )
        batch_index = torch.arange(scanpath.shape[0]).unsqueeze(1).expand_as(scanpath)
        scanpath_add1 = scanpath.clone()
        scanpath_add1[scanpath != SCANPATH_PADDING_VAL] += SCANPATH_PADDING_VAL
        p_embds_sp_order = merged_p_emb[
            batch_index, scanpath_add1
        ]  # [batch, max_sp_length, emb_dim]

        # note that p_embds_sp_order should never contain the embeddings of the <CLS> token

        # (4) combine the paragraph embeddings with features_by_sp_idx, features_by_word_idx
        if features_by_sp_idx is not None:
            p_embds_sp_order = torch.cat([p_embds_sp_order, features_by_sp_idx], dim=2)
        if features_by_word_idx is not None:
            p_embds_sp_order = torch.cat(
                [p_embds_sp_order, features_by_word_idx[batch_index, scanpath]], dim=2
            )  # we don't need scanpath_add1 here because no <s> token in the beginning

        # (5) combine parts into a sequence of embeddings to feed the lstm
        if (
            self.prediction_mode
            in BINARY_PARAGRAPH_ONLY_TASKS
            + BINARY_P_AND_Q_TASKS
            + REGRESSION_PARAGRAPH_ONLY_TASKS
        ):
            final_seq_embds = p_embds_sp_order
            final_seq_lens = (
                (scanpath != SCANPATH_PADDING_VAL).sum(dim=1).clone().detach()
            )

        else:
            raise NotImplementedError(
                f'Not implemented for prediction_mode: {self.prediction_mode}'
            )

        # pass through the LSTM layer
        sorted_lengths, indices = torch.sort(final_seq_lens, descending=True)
        final_seq_embds = final_seq_embds[
            indices
        ]  # reorder sequences according to the descending order of the lengths

        # Pass the entire sequence through the LSTM layer
        packed_final_seq_embds = nn.utils.rnn.pack_padded_sequence(
            input=final_seq_embds,
            lengths=sorted_lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=True,
        )

        packed_output, (hn, cn) = self.fse_lstm(packed_final_seq_embds)
        unpacked_output = nn.utils.rnn.unpack_sequence(packed_output)

        # unpacked output is a list of batch_size tensors,
        # each tensor is shaped [seq_len, no.directions*hidden_size]
        # create a tensor of shape [batch_size, no.directions*hidden_size]
        # by taking the mean of each element in the list
        lstm_mean_hidden = torch.stack([torch.mean(t, dim=0) for t in unpacked_output])

        # reorder the hidden states to the original order
        lstm_mean_hidden = lstm_mean_hidden[
            torch.argsort(indices)
        ]  # Tested. Reorders correctly

        logits = self.classifier_head(lstm_mean_hidden)

        return logits, lstm_mean_hidden

    def shared_step(
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            batch (tuple): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_

        Notes:
            - in input ids: 0 is for CLS, 2 is for SEP, 1 is for PAD
        """
        batch_data = self.unpack_batch(batch)
        assert batch_data.input_ids is not None, 'input_ids not in batch_dict'
        assert batch_data.input_masks is not None, 'input_masks not in batch_dict'
        assert batch_data.scanpath is not None, 'scanpath not in batch_dict'
        assert batch_data.fixation_features is not None, 'eyes_tensor not in batch_dict'
        assert batch_data.scanpath_pads is not None, 'scanpath_pads not in batch_dict'
        assert batch_data.eyes is not None, 'eye not in batch_dict'
        assert batch_data.p_input_ids is not None, 'p_input_ids not in batch_dict'

        # -----------------------------------------

        logits, _ = self.forward(
            input_ids=batch_data.input_ids,
            input_masks=batch_data.input_masks,
            p_input_ids=batch_data.p_input_ids,
            scanpath=batch_data.scanpath,
            fixation_features=batch_data.fixation_features,
            scanpath_pads=batch_data.scanpath_pads,
            eyes=batch_data.eyes,
        )

        labels = batch_data.labels

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, labels)

        return labels, loss, logits.squeeze()


def trim_scanpath_and_fixation_features(
    scanpath: torch.Tensor,
    fixation_features: torch.Tensor,
    scanpath_pads: torch.Tensor,
    max_scanpath_length: int,
):
    shortest_scanpath_pad = scanpath_pads.min()
    longest_batch_scanpath: int = int(max_scanpath_length - shortest_scanpath_pad)

    scanpath = scanpath[..., :longest_batch_scanpath]
    fixation_features = fixation_features[..., :longest_batch_scanpath, :]
    return scanpath, fixation_features
