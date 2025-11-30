"""This module contains the MAG module and the MAGRobertaModel class."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.roberta import (
    RobertaForSequenceClassification,
    RobertaModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaEmbeddings,
    RobertaLayer,
    RobertaPooler,
)
from transformers.models.xlm_roberta import (
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaClassificationHead,
    XLMRobertaEmbeddings,
    XLMRobertaLayer,
    XLMRobertaPooler,
)

from src.configs.constants import (
    BINARY_P_AND_Q_TASKS,
    BINARY_PARAGRAPH_ONLY_TASKS,
    REGRESSION_PARAGRAPH_ONLY_TASKS,
)
from src.configs.data import DataArgs
from src.configs.models.dl.MAG import MAG
from src.configs.trainers import TrainerDL
from src.models.base_model import register_model
from src.models.base_roberta import BaseMultiModalRoberta


@dataclass
class MultimodalConfig:
    dropout_prob: float
    beta_shift: float
    text_dim: int
    eyes_dim: int
    mag_injection_index: int


@register_model
class MAGModel(BaseMultiModalRoberta):
    """
    Model for Multiple Choice Question Answering and question prediction tasks.
    """

    def __init__(
        self,
        model_args: MAG,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

        eyes_projection_input_dim = model_args.eyes_dim

        self.multimodal_config = MultimodalConfig(
            dropout_prob=model_args.mag_dropout,
            beta_shift=model_args.mag_beta_shift,
            text_dim=model_args.text_dim,
            eyes_dim=eyes_projection_input_dim,
            mag_injection_index=model_args.mag_injection_index,
        )
        print(
            f'Injecting MAG at layer index {self.multimodal_config.mag_injection_index}'
        )
        if (
            self.prediction_mode
            in BINARY_PARAGRAPH_ONLY_TASKS
            + BINARY_P_AND_Q_TASKS
            + REGRESSION_PARAGRAPH_ONLY_TASKS
        ):
            if data_args.is_english:
                self.model = MAGRobertaForSequenceClassification.from_pretrained(
                    model_args.backbone,
                    num_labels=self.num_classes,
                    multimodal_config=self.multimodal_config,
                )
            else:
                self.model = XLMMAGRobertaForSequenceClassification.from_pretrained(
                    model_args.backbone,
                    num_labels=self.num_classes,
                    multimodal_config=self.multimodal_config,
                )
        else:
            raise ValueError(f'Untested prediction mode: {self.prediction_mode}.')

        if model_args.freeze:
            # Freeze all model parameters except specific ones
            for name, param in self.named_parameters():
                if name.startswith('model.roberta.encoder.mag') or name.startswith(
                    'model.classifier'
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.train()
        self.save_hyperparameters()


class MAGModule(nn.Module):
    """
    This class implements the Multimodal Attention Gate (MAG) module.
    """

    # Based on https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/modeling.py
    def __init__(self, hidden_size, beta_shift, dropout_prob, text_dim, eyes_dim):
        super().__init__()
        print(
            f'Initializing MAG with beta_shift:{beta_shift} hidden_prob:{dropout_prob}'
        )
        self.w_hv = nn.Linear(eyes_dim + text_dim, text_dim)
        self.w_v = nn.Linear(eyes_dim, text_dim)
        self.beta_shift = beta_shift

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text_embedding, gaze_features):
        # text_embedding: torch.Size([num_classes X batch_size, max_len, embed_dim])
        # eyes: torch.Size([num_classes X batch_size, max_len, num_features])
        eps = 1e-6

        # weight_v: torch.Size([num_classes X batch_size, max_len, embed_dim])
        weight_v = F.relu(self.w_hv(torch.cat((gaze_features, text_embedding), dim=-1)))

        h_m = weight_v * self.w_v(gaze_features)
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).type_as(hm_norm)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        threshhold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(threshhold.shape, requires_grad=True).type_as(threshhold)
        alpha = torch.min(threshhold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        zero_entries = (gaze_features == 0).all(dim=2)
        acoustic_vis_embedding[zero_entries] = 0

        return self.dropout(self.layer_norm(acoustic_vis_embedding + text_embedding))


class XLMMAGRobertaModel(XLMRobertaModel):
    """
    This class is a modified version of the RobertaModel class from the transformers library.
    It adds the MAG module to the forward pass.
    """

    def __init__(self, config, multimodal_config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config, multimodal_config=multimodal_config)

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        gaze_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]  # type: ignore
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            input_shape,  # type: ignore
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
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
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
        )

        encoder_outputs = self.encoder(
            embedding_output,
            gaze_features,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,  # type: ignore
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class XLMMAGRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    """
    This class is a modified version of the RobertaForSequenceClassification class
    from the transformers library.
    It adds the MAG module to the forward pass.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
    """

    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMMAGRobertaModel(
            config, multimodal_config, add_pooling_layer=False
        )
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        gaze_features: Optional[torch.Tensor] = None,
        gaze_positions: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        eye_token_type_ids: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids=input_ids,
            gaze_features=gaze_features,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)  # type: ignore
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):  # type: ignore
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  #  type: ignore
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # type: ignore
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class XLMRobertaEncoder(nn.Module):
    """
    This class is a modified version of the RobertaEncoder class from the transformers library.
    """

    def __init__(self, config, multimodal_config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.mag_injection_index = multimodal_config.mag_injection_index
        self.mag = MAGModule(
            hidden_size=config.hidden_size,
            beta_shift=multimodal_config.beta_shift,
            dropout_prob=multimodal_config.dropout_prob,
            text_dim=multimodal_config.text_dim,
            eyes_dim=multimodal_config.eyes_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gaze_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None | list = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    '`use_cache=True` is incompatible with gradient checkpointing. '
                    'Setting `use_cache=False`...'
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i == self.mag_injection_index and gaze_features is not None:
                hidden_states = self.mag(hidden_states, gaze_features)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            past_key_value,  # pylint: disable=cell-var-from-loop
                            output_attentions,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(  # type: ignore
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)  # type: ignore
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,  # type: ignore
            past_key_values=next_decoder_cache,  # type: ignore
            hidden_states=all_hidden_states,  # type: ignore
            attentions=all_self_attentions,  # type: ignore
            cross_attentions=all_cross_attentions,  # type: ignore
        )


class MAGRobertaModel(RobertaModel):
    """
    This class is a modified version of the RobertaModel class from the transformers library.
    It adds the MAG module to the forward pass.
    """

    def __init__(self, config, multimodal_config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config, multimodal_config=multimodal_config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        gaze_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, 'token_type_ids'):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]  # type: ignore
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            input_shape,  # type: ignore
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
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
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
        )

        encoder_outputs = self.encoder(
            embedding_output,
            gaze_features,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,  # type: ignore
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class MAGRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class is a modified version of the RobertaForSequenceClassification class
    from the transformers library.
    It adds the MAG module to the forward pass.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
    """

    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = MAGRobertaModel(
            config, multimodal_config, add_pooling_layer=False
        )
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        gaze_features: Optional[torch.Tensor] = None,
        gaze_positions: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        eye_token_type_ids: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids=input_ids,
            gaze_features=gaze_features,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)  # type: ignore
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):  # type: ignore
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  #  type: ignore
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # type: ignore
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    """
    This class is a modified version of the RobertaEncoder class from the transformers library.
    """

    def __init__(self, config, multimodal_config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.mag_injection_index = multimodal_config.mag_injection_index
        self.mag = MAGModule(
            hidden_size=config.hidden_size,
            beta_shift=multimodal_config.beta_shift,
            dropout_prob=multimodal_config.dropout_prob,
            text_dim=multimodal_config.text_dim,
            eyes_dim=multimodal_config.eyes_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gaze_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None | list = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    '`use_cache=True` is incompatible with gradient checkpointing. '
                    'Setting `use_cache=False`...'
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i == self.mag_injection_index and gaze_features is not None:
                hidden_states = self.mag(hidden_states, gaze_features)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            past_key_value,  # pylint: disable=cell-var-from-loop
                            output_attentions,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(  # type: ignore
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)  # type: ignore
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,  # type: ignore
            past_key_values=next_decoder_cache,  # type: ignore
            hidden_states=all_hidden_states,  # type: ignore
            attentions=all_self_attentions,  # type: ignore
            cross_attentions=all_cross_attentions,  # type: ignore
        )
