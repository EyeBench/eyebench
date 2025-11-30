import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.roberta import RobertaModel

from src.configs.constants import PredMode, TaskTypes
from src.configs.data import DataArgs
from src.configs.models.dl.PostFusion import PostFusion
from src.models.base_model import register_model
from src.models.base_roberta import BaseMultiModalRoberta


@register_model
class PostFusionModel(BaseMultiModalRoberta):
    def __init__(
        self,
        model_args: PostFusion,
        trainer_args,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
        self.add_question = data_args.task == PredMode.RC
        self.preorder = model_args.preorder
        print(f'preorder: {self.preorder}')
        self.d_eyes = model_args.fixation_dim
        self.d_conv = model_args.text_dim // 2
        self.sep_token_id = self.model_args.sep_token_id
        self.use_attn_mask = model_args.use_attn_mask

        self.fixation_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.d_eyes,
                out_channels=self.d_conv // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm1d(num_features=self.d_conv // 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.d_conv // 2,
                out_channels=self.d_conv,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm1d(num_features=self.d_conv),
            torch.nn.ReLU(),
        )

        self.roberta = RobertaModel.from_pretrained(
            pretrained_model_name_or_path=model_args.backbone
        )

        self.cross_att_eyes_p = torch.nn.MultiheadAttention(
            embed_dim=self.d_conv,
            num_heads=1,
            dropout=model_args.cross_attention_dropout,
            kdim=model_args.text_dim,
            vdim=model_args.text_dim,
            batch_first=True,
        )

        self.cross_att_agg_eyes_q = torch.nn.MultiheadAttention(
            embed_dim=model_args.text_dim,
            num_heads=1,
            dropout=model_args.cross_attention_dropout,
            kdim=model_args.text_dim,
            vdim=model_args.text_dim,
            batch_first=True,
        )

        self.project_to_text_dim = torch.nn.Sequential(
            torch.nn.Dropout(p=model_args.eye_projection_dropout),
            torch.nn.Linear(
                in_features=model_args.text_dim,
                out_features=model_args.text_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=model_args.eye_projection_dropout),
            torch.nn.Linear(
                in_features=model_args.text_dim,
                out_features=model_args.text_dim,
            ),
            torch.nn.LeakyReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model_args.text_dim, out_features=model_args.text_dim // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=model_args.text_dim // 2, out_features=self.num_classes
            ),
        )
        self.train()
        if model_args.freeze:
            # Freeze all model parameters except specific ones
            for name, param in self.named_parameters():
                if (
                    name.startswith('cross_att_eyes_p')
                    or name.startswith('cross_att_agg_eyes_q')
                    or name.startswith('project_to_text_dim')
                    or name.startswith('classifier')
                    or name.startswith('fixation_encoder')
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.train()
        self.save_hyperparameters()

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        labels = batch_data.labels

        if self.model_args.use_fixation_report:
            assert batch_data.scanpath_pads is not None
            assert batch_data.scanpath is not None
            assert batch_data.fixation_features is not None
            assert batch_data.input_masks is not None
            assert batch_data.grouped_inversions is not None

            longest_batch_scanpath = (
                self.max_scanpath_length - batch_data.scanpath_pads.min()
            )

            eye_text_attn_mask = (
                batch_data.grouped_inversions[..., :longest_batch_scanpath, :]
                if self.use_attn_mask
                else None
            )

            gaze_features = batch_data.fixation_features[
                ..., :longest_batch_scanpath, :
            ]

        else:
            assert batch_data.eyes is not None
            assert batch_data.input_ids is not None
            gaze_features = batch_data.eyes

        # permute the gaze_features to (batch_size, d_eyes, max_eye_len)
        gaze_features = gaze_features.permute(0, 2, 1)

        # convert eye_text_attn_mask to bool
        if eye_text_attn_mask is not None:
            eye_text_attn_mask += 1
            eye_text_attn_mask = eye_text_attn_mask.bool()

        logits, x = self(
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.input_masks,
            gaze_features=gaze_features,
            eye_text_attn_mask=eye_text_attn_mask,
        )

        if self.task == TaskTypes.REGRESSION:
            labels = labels.squeeze().float()
            logits = logits.squeeze()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, labels)

        return labels, loss, logits.squeeze()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gaze_features: torch.Tensor,
        eye_text_attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_fixations = self.fixation_encoder(
            gaze_features
        )  # (batch_size, d_conv, max_eye_len)
        encoded_fixations = encoded_fixations.permute(
            0, 2, 1
        )  # (batch_size, max_eye_len, d_conv)

        encoded_word_seq = self.roberta(
            input_ids, attention_mask
        ).last_hidden_state  # (batch_size, max_seq_len, text_dim) # type: ignore

        p_embds, q_embds, p_masks, q_masks = self.split_context_embds_batched(
            encoded_word_seq, input_ids
        )

        eye_text_attn = self.cross_att_eyes_p(
            query=encoded_fixations,
            key=p_embds,
            value=p_embds,
            attn_mask=eye_text_attn_mask,
            need_weights=False,
        )[0]  # (batch_size, max_eye_len, text_dim)

        # concat eye_text_attn with encoded_fixations
        eye_text = torch.cat(
            (eye_text_attn, encoded_fixations), dim=2
        )  # (batch_size, max_eye_len, d_conv * 2)
        # mean pool the question embeddings from tokens to a single vector
        # note that the questions are pad with zeros so we can't just take the mean
        # but rather use sum and divide by the number of non-zero elements
        if self.add_question:
            q_embds_mean_pool = q_embds.sum(dim=1) / q_masks.sum(dim=1).unsqueeze(1).to(
                q_embds.device
            )
            x = self.agg_fixation_by_question(eye_text, q_embds_mean_pool)
        else:
            x = eye_text.mean(dim=1)
        x = x.squeeze()
        output = self.classifier(x)
        # output = self.score_answers(x, a_embeds)

        return output, x

    def split_context_embeds(
        self,
        encoded_word_seq: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            a = torch.zeros_like(q).unsqueeze(0)
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
            q = p.mean(dim=0).unsqueeze(0)  # shape: (1, text_dim)
            # q = q[1:, :].mean(dim=0)
            a = p.mean(dim=0).unsqueeze(0)  # shape: (1, text_dim)

        # a1 = a1[1:, :].mean(dim=0)
        # a2 = a2[1:, :].mean(dim=0)
        # a3 = a3[1:, :].mean(dim=0)
        # a4 = a4[1:, :].mean(dim=0)
        # a_embeds = torch.stack((a1, a2, a3, a4), dim=0)
        # a_embeds = a[1:, :].mean(dim=0)
        return p, q, a

    def split_context_embds_batched(self, encoded_word_seq, input_ids):
        p_embds_batches, q_embds_batches, _ = [], [], []
        # Process each batch separately
        for ewsb, iib in zip(encoded_word_seq, input_ids):
            p_embds, q_embds, _ = self.split_context_embeds(
                encoded_word_seq=ewsb, input_ids=iib
            )

            p_embds_batches.append(p_embds)
            q_embds_batches.append(q_embds)
            # a_embeds_batches.append(a_embeds)

        # pad the embeddings to the maximum sequence length of each list
        p_max_len = self.actual_max_needed_len
        q_max_len = max([q.shape[0] for q in q_embds_batches])
        # a_max_len = max([a.shape[0] for a in a_embeds_batches])

        # create the masks of where the padding is
        p_masks = torch.stack(
            [
                torch.cat([torch.ones(p.shape[0]), torch.zeros(p_max_len - p.shape[0])])
                for p in p_embds_batches
            ],
            dim=0,
        )
        q_masks = torch.stack(
            [
                torch.cat([torch.ones(q.shape[0]), torch.zeros(q_max_len - q.shape[0])])
                for q in q_embds_batches
            ],
            dim=0,
        )
        # a_masks = torch.stack(
        #     [
        #         torch.cat([torch.ones(a.shape[0]), torch.zeros(a_max_len - a.shape[0])])
        #         for a in a_embeds_batches
        #     ],
        #     dim=0,
        # )

        # add padding
        p_embds_batches = [
            F.pad(p, (0, 0, 0, p_max_len - p.shape[0])) for p in p_embds_batches
        ]
        q_embds_batches = [
            F.pad(q, (0, 0, 0, q_max_len - q.shape[0])) for q in q_embds_batches
        ]
        # a_embeds_batches = [
        #     F.pad(a, (0, 0, 0, a_max_len - a.shape[0])) for a in a_embeds_batches
        # ]

        # Concatenate the results back together
        p_embds = torch.stack(p_embds_batches, dim=0)
        q_embds = torch.stack(q_embds_batches, dim=0)
        # a_embeds = torch.stack(a_embeds_batches, dim=0)

        return p_embds, q_embds, p_masks, q_masks

    def agg_fixation_by_question(self, eye_text, q_embds):
        # run eye_text_attn through a linear layer so it matches the shape of q_embds
        eye_text_attn = self.project_to_text_dim(eye_text)
        return self.cross_att_agg_eyes_q(
            query=q_embds.unsqueeze(1),
            key=eye_text_attn,
            value=eye_text_attn,
            need_weights=False,
        )[0]

    def score_answers(
        self, x, a_embeds
    ) -> torch.Tensor:  # a_embeds: (2 X 4 X 1024) x: (2 X 1024)
        return torch.bmm(x, a_embeds.transpose(1, 2)).squeeze(1)
