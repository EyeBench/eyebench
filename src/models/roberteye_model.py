"""roberteye.py"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.roberta import (
    RobertaForSequenceClassification,
    RobertaModel,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaPooler,
    create_position_ids_from_input_ids,
)
from transformers.models.xlm_roberta import (
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaClassificationHead,
    XLMRobertaEmbeddings,
    XLMRobertaEncoder,
    XLMRobertaPooler,
)

from src.configs.constants import (
    BINARY_P_AND_Q_TASKS,
    BINARY_PARAGRAPH_ONLY_TASKS,
    REGRESSION_PARAGRAPH_ONLY_TASKS,
)
from src.configs.data import DataArgs
from src.configs.models.dl.RoBERTeye import RoberteyeArgs
from src.configs.trainers import TrainerDL
from src.models.base_model import register_model
from src.models.base_roberta import BaseMultiModalRoberta
from src.models.mag_model import MAGModule


@dataclass
class MultimodalConfig:
    text_dim: int
    eyes_dim: int
    dropout: float
    eye_projection_MAGModule: bool


@register_model
class Roberteye(BaseMultiModalRoberta):
    """
    Model for Multiple Choice Question Answering and question prediction tasks.
    """

    def __init__(
        self,
        model_args: RoberteyeArgs,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )
        print(f'Is training: {model_args.is_training}')

        if model_args.use_fixation_report:
            eyes_projection_input_dim = model_args.fixation_dim
        else:
            eyes_projection_input_dim = model_args.eyes_dim

        assert isinstance(eyes_projection_input_dim, int)

        self.multimodal_config = MultimodalConfig(
            text_dim=model_args.text_dim,
            eyes_dim=eyes_projection_input_dim,
            dropout=model_args.eye_projection_dropout,
            eye_projection_MAGModule=model_args.eye_projection_MAGModule,
        )

        if (
            self.prediction_mode
            in []
            + BINARY_PARAGRAPH_ONLY_TASKS
            + BINARY_P_AND_Q_TASKS
            + REGRESSION_PARAGRAPH_ONLY_TASKS
        ):
            if model_args.is_training:
                if data_args.is_english:
                    model = RobertEyeForSequenceClassification.from_pretrained(
                        model_args.backbone,
                        num_labels=self.num_classes,
                        multimodal_config=self.multimodal_config,
                    )
                else:
                    model = XLMRobertEyeForSequenceClassification.from_pretrained(
                        model_args.backbone,
                        num_labels=self.num_classes,
                        multimodal_config=self.multimodal_config,
                    )
                model = adjust_model_for_eyes(
                    model,
                    eye_token_id=model_args.eye_token_id,
                    sep_token_id=model_args.sep_token_id,
                    token_type_num=model_args.token_type_num,
                )
                self.model = model
                self.train()
            else:
                if data_args.is_english:
                    robertaconfig = RobertaConfig.from_pretrained(
                        model_args.backbone,
                        vocab_size=model_args.vocab_size,
                        type_vocab_size=model_args.token_type_num,
                        num_labels=self.num_classes,
                    )

                    self.model = RobertEyeForSequenceClassification(
                        config=robertaconfig,
                        multimodal_config=self.multimodal_config,
                    )
                else:
                    robertaconfig = XLMRobertaConfig.from_pretrained(
                        model_args.backbone,
                        vocab_size=model_args.vocab_size,
                        type_vocab_size=model_args.token_type_num,
                        num_labels=self.num_classes,
                    )

                    self.model = XLMRobertEyeForSequenceClassification(
                        config=robertaconfig,
                        multimodal_config=self.multimodal_config,
                    )

        else:
            raise ValueError(
                f'Invalid combination: prediction_mode - {self.prediction_mode}, '
            )

        if model_args.freeze:
            # Freeze all model parameters except specific ones
            for name, param in self.named_parameters():
                if (
                    name.startswith('model.roberta.embeddings.eye_projection')
                    or name.startswith(
                        'model.roberta.embeddings.eye_position_embeddings',
                    )
                    or name.startswith(
                        'model.roberta.embeddings.EyeProjectionMAGModule',
                    )
                    or name.startswith('model.roberta.embeddings.token_type_embeddings')
                    or name.startswith('model.classifier')
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # self.model = torch.compile(self.model, fullgraph=True)
        self.train()
        self.save_hyperparameters()


class XLMRoberteyeEmbeddings(XLMRobertaEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    Based on https://github.com/uclanlp/visualbert/tree/master/visualbert
    """

    def __init__(self, config: XLMRobertaConfig, multimodal_config: MultimodalConfig):
        super().__init__(config)
        # Token type and position embedding for eye features
        self.eye_position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.eye_position_embeddings.weight.data = nn.Parameter(
            self.position_embeddings.weight.data.clone(),
            requires_grad=True,
        )
        projection_dropout = multimodal_config.dropout
        self.eye_projection = nn.Sequential(
            nn.Linear(
                in_features=multimodal_config.eyes_dim,
                out_features=config.hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(p=projection_dropout),
            nn.Linear(
                in_features=config.hidden_size // 2,
                out_features=config.hidden_size,
            ),
        )

        self.project_eyes_with_MAG = multimodal_config.eye_projection_MAGModule
        if self.project_eyes_with_MAG:
            self.EyeProjectionMAGModule = MAGModule(
                hidden_size=config.hidden_size,
                beta_shift=0.5,
                dropout_prob=projection_dropout,
                text_dim=config.hidden_size,
                eyes_dim=multimodal_config.eyes_dim,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
        eye_embeds: torch.Tensor | None = None,
        eye_token_type_ids: torch.Tensor | None = None,
        eye_position_ids: torch.Tensor | None = None,
        eye_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids,
                    self.padding_idx,
                    past_key_values_length,
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds,
                )

        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('Either input_ids or inputs_embeds must be provided.')

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids,
        # solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0],
                    seq_length,
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Eye movements addition
        if eye_embeds is not None:
            if eye_token_type_ids is None:
                eye_token_type_ids = torch.ones(
                    eye_embeds.size()[:-1],
                    dtype=torch.long,
                    device=self.position_ids.device,
                )
            assert eye_positions is not None
            if self.project_eyes_with_MAG:
                # Initialize an empty tensor to store the results
                batch_size, seq_len, num_positions = eye_positions.shape
                embed_dim = inputs_embeds.shape[-1]
                average_embeddings = torch.zeros(
                    (batch_size, seq_len, embed_dim),
                    device=inputs_embeds.device,
                )

                # Compute the average embeddings
                for i in range(batch_size):
                    for j in range(seq_len):
                        # Get the valid positions (ignoring -1)
                        valid_positions = eye_positions[i, j][eye_positions[i, j] != -1]
                        if len(valid_positions) > 0:
                            # Gather the word embeddings at the valid positions
                            selected_embeddings = inputs_embeds[i, valid_positions]
                            # Compute the average and store it
                            average_embeddings[i, j] = selected_embeddings.mean(dim=0)

                eye_embeds = self.EyeProjectionMAGModule(
                    text_embedding=average_embeddings,
                    gaze_features=eye_embeds,
                )
            else:
                eye_embeds = self.eye_projection(eye_embeds)
            eye_token_type_embeddings = self.token_type_embeddings(eye_token_type_ids)

            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

            dtype = token_type_embeddings.dtype
            eyes_text_alignment_mask = (eye_positions != -1).long()
            # Get rid of the -1.
            eye_positions = eyes_text_alignment_mask * eye_positions

            # Batch x image_length x alignment length x dim
            eye_position_embeddings = self.position_embeddings(eye_positions)
            eye_position_embeddings *= eyes_text_alignment_mask.to(
                dtype=dtype,
            ).unsqueeze(-1)
            eye_position_embeddings = eye_position_embeddings.sum(2)

            # We want to average along the alignment_number dimension.
            eyes_text_alignment_mask = eyes_text_alignment_mask.to(dtype=dtype).sum(2)

            if (eyes_text_alignment_mask == 0).sum() != 0:
                eyes_text_alignment_mask[eyes_text_alignment_mask == 0] = (
                    1  # Avoid divide by zero error
                )
                # print(
                #     "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                #     " error."
                # )
            eye_position_embeddings = (
                eye_position_embeddings / eyes_text_alignment_mask.unsqueeze(-1)
            )

            # visual_position_ids = torch.zeros(
            #     *eye_embeds.size()[:-1], dtype=torch.long, device=eye_embeds.device
            # )

            # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
            if eye_position_embeddings.size(1) != eye_embeds.size(1):
                if eye_position_embeddings.size(1) < eye_embeds.size(1):
                    raise ValueError(
                        f'Visual position embeddings length: {eye_position_embeddings.size(1)} '
                        f'should be the same as `eye_embeds` length: {eye_embeds.size(1)}',
                    )
                eye_position_embeddings = eye_position_embeddings[
                    :,
                    : eye_embeds.size(1),
                    :,
                ]

            # eye_position_embeddings = eye_position_embeddings + self.eye_position_embeddings(
            #     visual_position_ids
            # )

            # if eye_position_ids is None:
            #     eye_position_ids = create_position_ids_from_input_ids(
            #         eye_positions,
            #         self.padding_idx,
            #         past_key_values_length,
            #     )
            # eye_position_embeddings = self.eye_position_embeddings(eye_position_ids)

            final_eye_embeds = (
                eye_embeds + eye_position_embeddings + eye_token_type_embeddings
            )

            cls_token, rest_embedding_output = (
                embeddings[:, 0:1, :],
                embeddings[:, 1:, :],
            )
            # Concatenate the CLS token, eye, and the rest of the embedding_output
            embeddings = torch.cat(
                (cls_token, final_eye_embeds, rest_embedding_output),
                dim=1,
            )
            # Final format: CLS EYES EYE_TOKEN SEP_TOKEN REST_OF_THE_TEXT
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class XLMRoBERTeyeEncoderModel(XLMRobertaModel):
    """
    This class is a modified version of the RobertaModel class from the transformers library.
    It adds the MAG module to the forward pass.
    """

    def __init__(
        self,
        config: XLMRobertaConfig,
        multimodal_config: MultimodalConfig,
        add_pooling_layer: bool = True,
    ):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = XLMRoberteyeEmbeddings(config, multimodal_config)
        self.encoder = XLMRobertaEncoder(config)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        eye_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape
        `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape
        `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with
        each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape
            `(batch_size, 1)` instead of all `decoder_input_ids` of
            shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding (see `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time',
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        if gaze_features is not None:
            input_shape = torch.Size(
                (input_shape[0], input_shape[1] + gaze_features.size()[1]),
            )

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=self.device,
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                token_seq_length = input_ids.size()[1]
                buffered_token_type_ids = self.embeddings.token_type_ids[
                    :, :token_seq_length
                ]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, token_seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.device
                )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape,
                    device=self.device,
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask,
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            eye_token_type_ids=eye_token_type_ids,
            eye_embeds=gaze_features,
            eye_positions=gaze_positions,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class XLMRobertEyeForSequenceClassification(XLMRobertaForSequenceClassification):
    """
    This class is a modified version of the RobertaForSequenceClassification class
    from the transformers library.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
    """

    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config: XLMRobertaConfig, multimodal_config: MultimodalConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRoBERTeyeEncoderModel(
            config,
            multimodal_config,
            add_pooling_layer=False,
        )
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # self.roberta = torch.compile(self.roberta, fullgraph=True)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        eye_token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.roberta(
            input_ids=input_ids,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            eye_token_type_ids=eye_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoberteyeEmbeddings(RobertaEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    Based on https://github.com/uclanlp/visualbert/tree/master/visualbert
    """

    def __init__(self, config: RobertaConfig, multimodal_config: MultimodalConfig):
        super().__init__(config)
        # Token type and position embedding for eye features
        self.eye_position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.eye_position_embeddings.weight.data = nn.Parameter(
            self.position_embeddings.weight.data.clone(),
            requires_grad=True,
        )
        projection_dropout = multimodal_config.dropout
        self.eye_projection = nn.Sequential(
            nn.Linear(
                in_features=multimodal_config.eyes_dim,
                out_features=config.hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(p=projection_dropout),
            nn.Linear(
                in_features=config.hidden_size // 2,
                out_features=config.hidden_size,
            ),
        )

        self.project_eyes_with_MAG = multimodal_config.eye_projection_MAGModule
        if self.project_eyes_with_MAG:
            self.EyeProjectionMAGModule = MAGModule(
                hidden_size=config.hidden_size,
                beta_shift=0.5,
                dropout_prob=projection_dropout,
                text_dim=config.hidden_size,
                eyes_dim=multimodal_config.eyes_dim,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
        eye_embeds: torch.Tensor | None = None,
        eye_token_type_ids: torch.Tensor | None = None,
        eye_position_ids: torch.Tensor | None = None,
        eye_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids,
                    self.padding_idx,
                    past_key_values_length,
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds,
                )

        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('Either input_ids or inputs_embeds must be provided.')

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids,
        # solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0],
                    seq_length,
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Eye movements addition
        if eye_embeds is not None:
            if eye_token_type_ids is None:
                eye_token_type_ids = torch.ones(
                    eye_embeds.size()[:-1],
                    dtype=torch.long,
                    device=self.position_ids.device,
                )
            assert eye_positions is not None
            if self.project_eyes_with_MAG:
                # Initialize an empty tensor to store the results
                batch_size, seq_len, num_positions = eye_positions.shape
                embed_dim = inputs_embeds.shape[-1]
                average_embeddings = torch.zeros(
                    (batch_size, seq_len, embed_dim),
                    device=inputs_embeds.device,
                )

                # Compute the average embeddings
                for i in range(batch_size):
                    for j in range(seq_len):
                        # Get the valid positions (ignoring -1)
                        valid_positions = eye_positions[i, j][eye_positions[i, j] != -1]
                        if len(valid_positions) > 0:
                            # Gather the word embeddings at the valid positions
                            selected_embeddings = inputs_embeds[i, valid_positions]
                            # Compute the average and store it
                            average_embeddings[i, j] = selected_embeddings.mean(dim=0)

                eye_embeds = self.EyeProjectionMAGModule(
                    text_embedding=average_embeddings,
                    gaze_features=eye_embeds,
                )
            else:
                eye_embeds = self.eye_projection(eye_embeds)
            eye_token_type_embeddings = self.token_type_embeddings(eye_token_type_ids)

            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

            dtype = token_type_embeddings.dtype
            eyes_text_alignment_mask = (eye_positions != -1).long()
            # Get rid of the -1.
            eye_positions = eyes_text_alignment_mask * eye_positions

            # Batch x image_length x alignment length x dim
            eye_position_embeddings = self.position_embeddings(eye_positions)
            eye_position_embeddings *= eyes_text_alignment_mask.to(
                dtype=dtype,
            ).unsqueeze(-1)
            eye_position_embeddings = eye_position_embeddings.sum(2)

            # We want to average along the alignment_number dimension.
            eyes_text_alignment_mask = eyes_text_alignment_mask.to(dtype=dtype).sum(2)

            if (eyes_text_alignment_mask == 0).sum() != 0:
                eyes_text_alignment_mask[eyes_text_alignment_mask == 0] = (
                    1  # Avoid divide by zero error
                )
                # print(
                #     "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                #     " error."
                # )
            eye_position_embeddings = (
                eye_position_embeddings / eyes_text_alignment_mask.unsqueeze(-1)
            )

            # visual_position_ids = torch.zeros(
            #     *eye_embeds.size()[:-1], dtype=torch.long, device=eye_embeds.device
            # )

            # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
            if eye_position_embeddings.size(1) != eye_embeds.size(1):
                if eye_position_embeddings.size(1) < eye_embeds.size(1):
                    raise ValueError(
                        f'Visual position embeddings length: {eye_position_embeddings.size(1)} '
                        f'should be the same as `eye_embeds` length: {eye_embeds.size(1)}',
                    )
                eye_position_embeddings = eye_position_embeddings[
                    :,
                    : eye_embeds.size(1),
                    :,
                ]

            # eye_position_embeddings = eye_position_embeddings + self.eye_position_embeddings(
            #     visual_position_ids
            # )

            # if eye_position_ids is None:
            #     eye_position_ids = create_position_ids_from_input_ids(
            #         eye_positions,
            #         self.padding_idx,
            #         past_key_values_length,
            #     )
            # eye_position_embeddings = self.eye_position_embeddings(eye_position_ids)

            final_eye_embeds = (
                eye_embeds + eye_position_embeddings + eye_token_type_embeddings
            )

            cls_token, rest_embedding_output = (
                embeddings[:, 0:1, :],
                embeddings[:, 1:, :],
            )
            # Concatenate the CLS token, eye, and the rest of the embedding_output
            embeddings = torch.cat(
                (cls_token, final_eye_embeds, rest_embedding_output),
                dim=1,
            )
            # Final format: CLS EYES EYE_TOKEN SEP_TOKEN REST_OF_THE_TEXT
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoBERTeyeEncoderModel(RobertaModel):
    """
    This class is a modified version of the RobertaModel class from the transformers library.
    It adds the MAG module to the forward pass.
    """

    def __init__(
        self,
        config: RobertaConfig,
        multimodal_config: MultimodalConfig,
        add_pooling_layer: bool = True,
    ):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = RoberteyeEmbeddings(config, multimodal_config)
        self.encoder = RobertaEncoder(config)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        eye_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape
        `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape
        `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with
        each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape
            `(batch_size, 1)` instead of all `decoder_input_ids` of
            shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding (see `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time',
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        if gaze_features is not None:
            input_shape = torch.Size(
                (input_shape[0], input_shape[1] + gaze_features.size()[1]),
            )

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=self.device,
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                token_seq_length = input_ids.size()[1]
                buffered_token_type_ids = self.embeddings.token_type_ids[
                    :, :token_seq_length
                ]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, token_seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.device
                )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape,
                    device=self.device,
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask,
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            eye_token_type_ids=eye_token_type_ids,
            eye_embeds=gaze_features,
            eye_positions=gaze_positions,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertEyeForSequenceClassification(RobertaForSequenceClassification):
    """
    This class is a modified version of the RobertaForSequenceClassification class
    from the transformers library.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
    """

    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config: RobertaConfig, multimodal_config: MultimodalConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RoBERTeyeEncoderModel(
            config,
            multimodal_config,
            add_pooling_layer=False,
        )
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # self.roberta = torch.compile(self.roberta, fullgraph=True)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        eye_token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.roberta(
            input_ids=input_ids,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            eye_token_type_ids=eye_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def adjust_model_for_eyes(
    model: RobertEyeForSequenceClassification | XLMRobertEyeForSequenceClassification,
    eye_token_id: int,
    sep_token_id: int,
    token_type_num: int,  # defaults to 2 in RoBERTEyeArgs (1 for text and 1 for eyes)
) -> RobertEyeForSequenceClassification | XLMRobertEyeForSequenceClassification:
    model.config.vocab_size += 1
    # Add token_type+1 (for eye token id (=1) and token_embedding+1 (for Eye SEP)
    # Add 1 to the vocab size for the eye token
    # https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    model.resize_token_embeddings(model.config.vocab_size)

    # Itialize the eye token embedding to the SEP token embedding
    with torch.no_grad():
        model.roberta.embeddings.word_embeddings.weight[eye_token_id] = (
            model.roberta.embeddings.word_embeddings.weight[sep_token_id]
            .detach()
            .clone()
        )

    model.config.type_vocab_size = token_type_num

    single_emb: nn.Embedding = model.roberta.embeddings.token_type_embeddings

    model.roberta.embeddings.token_type_embeddings = nn.Embedding(
        model.config.type_vocab_size,
        single_emb.embedding_dim,
    )

    # https://github.com/huggingface/transformers/issues/1538#issuecomment-570260748
    model.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(
        single_emb.weight.repeat([model.config.type_vocab_size, 1])
    )
    return model
