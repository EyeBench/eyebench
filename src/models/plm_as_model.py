"""PLM-AS model for fixation sequence encoding and classification."""

import string
from typing import List

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.configs.constants import (
    SCANPATH_PADDING_VAL,
)
from src.configs.data import DataArgs
from src.configs.models.dl.PLMAS import PLMASArgs
from src.configs.trainers import TrainerDL
from src.models.base_model import BaseModel, register_model


@register_model
class PLMASModel(BaseModel):
    def __init__(
        self,
        model_args: PLMASArgs,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ):
        super().__init__(
            model_args=model_args, trainer_args=trainer_args, data_args=data_args
        )

        self.model_args = model_args
        self.backbone = model_args.backbone
        self.freeze_bert = model_args.freeze

        self.fast_tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.pad_token_id = self.fast_tokenizer.pad_token_id
        # Cache tokenizer with add_prefix_space for scanpath processing
        self.fast_tokenizer_prefix = AutoTokenizer.from_pretrained(
            self.backbone, add_prefix_space=True
        )

        # ? self.preorder = False

        self.classifier_head = nn.Linear(model_args.text_dim, self.num_classes)
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

        # create fse_lstm
        self.fse_lstm = nn.GRU(
            input_size=self.bert_dim,
            hidden_size=self.bert_dim,
            num_layers=self.model_args.lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.model_args.lstm_dropout,
        )

        self.train()
        self.save_hyperparameters()

    def fixation_sequence_encoder(
        self,
        sn_emd,
        sn_mask,
        word_ids_sn,
        sn_word_len,
        sp_emd,
        sp_pos,
        sp_fix_dur,
        sp_landing_pos,
        word_ids_sp,
        scanpath,
        features_by_sp_idx=None,
        features_by_word_idx=None,
    ):
        """A LSTM based encoder for the fixation sequence (scanpath)
        Args:
            sp_emd (torch.Tensor): A tensor containing the text input_ids ordered according to the scanpath
            sp_pos (torch.Tensor): The word index of each fixation in the scanpath (the word the fixation is on)
            sp_fix_dur (torch.Tensor): The total fixation duration of each word in the scanpath (fixation)
            sp_landing_pos (torch.Tensor): The landing position of each word in the scanpath (fixation)
            sp_mask (torch.Tensor): The mask for the scanpath
            word_ids_sp (torch.Tensor): The word index of each input_id in the scanpath
            scanpath (torch.Tensor): The scanpath tensor

        """

        # used for computing sp_merged_word_mask
        # x = self.bert_encoder.embeddings.word_embeddings(sp_emd)
        # x[sp_emd == self.pad_token_id] = 0
        # # Pool bert subword to word level for English corpus
        # _, sp_merged_word_mask = self.pool_subword_to_word(
        #     x, word_ids_sp, target='sp', pool_method='sum'
        # )

        with torch.no_grad():
            outputs = self.bert_encoder(input_ids=sn_emd, attention_mask=sn_mask)
        #  Make the embedding of the <pad> token to be zeros
        outputs.last_hidden_state[sn_emd == self.pad_token_id] = 0
        # Pool bert subword to word level for english corpus
        merged_word_emb, sn_mask_word = pool_subword_to_word(
            outputs.last_hidden_state,
            word_ids_sn,
            target='sn',
            max_seq_len=self.actual_max_needed_len,
            bert_dim=self.bert_dim,
            pool_method='sum',
        )
        batch_index = torch.arange(scanpath.shape[0]).unsqueeze(1).expand_as(scanpath)
        scanpath_add1 = scanpath.clone()
        scanpath_add1[scanpath != SCANPATH_PADDING_VAL] += SCANPATH_PADDING_VAL
        x = merged_word_emb[
            batch_index, scanpath_add1
        ]  # [batch, max_sp_length, emb_dim], word_emb_sn

        if features_by_sp_idx is not None:
            x = torch.cat([x, features_by_sp_idx], dim=2)
        if features_by_word_idx is not None:
            x = torch.cat(
                [x, features_by_word_idx[batch_index, scanpath]], dim=2
            )  # we don't need scanpath_add1 here because no <s> token in the beginning

        # pass through the LSTM layer
        sorted_lengths, indices = torch.sort(
            (scanpath != SCANPATH_PADDING_VAL).sum(dim=1), descending=True
        )
        x = x[
            indices
        ]  # reorder sequences according to the descending order of the lengths

        # Pass the entire sequence through the LSTM layer
        packed_x = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=sorted_lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=True,
        )

        # set h0 as [CLS] token embedding of each sequence
        h0 = outputs.last_hidden_state[indices, 0]

        # fit h0 and c0 to the shape of the LSTM
        # h0 is [batch, hidden_size] where h0[i] is the initial hidden state for the sequence i
        h0 = h0.unsqueeze(0).repeat(self.fse_lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        # ensure contiguous
        h0 = h0.contiguous()
        c0 = c0.contiguous()

        # * c0 isn't used when using GRU

        packed_output, ht = self.fse_lstm(packed_x, h0)
        lstm_last_hidden = ht[-1].squeeze(
            1
        )  # Take the hidden state of the last LSTM layer.

        # reorder the hidden states to the original order
        lstm_last_hidden = lstm_last_hidden[
            torch.argsort(indices)
        ]  # Tested. Reorders correctly

        # Clear unused tensors to free memory
        del outputs, packed_x, packed_output, ht, x, merged_word_emb

        return lstm_last_hidden  # Take the hidden state of the 8th LSTM layer.

    def forward(
        self,
        sn_emd,
        sn_mask,
        word_ids_sn,
        sn_word_len,
        sp_emd,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
        sp_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_fix_dur,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_landing_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        word_ids_sp,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
        scanpath,  # (Batch, Maximum length of the scanpath in WORDS)
        features_by_sp_idx=None,
        features_by_word_idx=None,
    ):
        assert (
            sn_emd[:, 0].sum().item() == 0
        )  # The CLS token is always present first (and 0 in roberta)

        fse_output = self.fixation_sequence_encoder(
            sn_emd=sn_emd,
            sn_mask=sn_mask,
            word_ids_sn=word_ids_sn,
            sn_word_len=sn_word_len,
            sp_emd=sp_emd,
            sp_pos=sp_pos,
            sp_fix_dur=sp_fix_dur,
            sp_landing_pos=sp_landing_pos,
            word_ids_sp=word_ids_sp,
            scanpath=scanpath,
            features_by_sp_idx=features_by_sp_idx,
            features_by_word_idx=features_by_word_idx,
        )  # [batch, step, dec_o_dim]

        pred = self.classifier_head(fse_output)

        return pred, fse_output

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

        shortest_scanpath_pad = batch_data.scanpath_pads.min()
        longest_batch_scanpath: int = int(
            self.max_scanpath_length - shortest_scanpath_pad
        )

        scanpath = batch_data.scanpath[..., :longest_batch_scanpath]
        fixation_features = batch_data.fixation_features[
            ..., :longest_batch_scanpath, :
        ]

        # scanpath masks
        sp_masks = torch.ones_like(scanpath)
        sp_masks[scanpath == self.pad_token_id] = 0

        decoded_to_txt_input_ids = self.fast_tokenizer.batch_decode(
            batch_data.input_ids, return_tensors='pt'
        )

        word_ids_sn = align_word_ids_with_input_ids(
            tokenizer=self.fast_tokenizer,
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        # in the decoded texts, space between <pad><pad>, <pad><s>, etc.
        decoded_to_txt_input_ids = list(
            map(
                lambda x: x.replace('<', ' <').split(' ')[1:],
                decoded_to_txt_input_ids,
            )
        )

        sn_word_len = get_sn_word_lens(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        word_ids_sp, sp_input_ids = calc_sp_word_input_ids(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
            roberta_tokenizer_prefix_space=self.fast_tokenizer_prefix,
            scanpath=scanpath,
        )

        sp_pos, sp_fix_dur, sp_landing_pos = eyettention_legacy_code(
            scanpath=scanpath,
            fixation_features=fixation_features,
        )

        sn_embed = batch_data.input_ids
        sn_mask = batch_data.input_masks
        # if the second dimension of the scanpath is more than the maximum context length (of self.bert_encoder), cut it and notify
        bert_encoder_max_len = self.bert_encoder.config.max_position_embeddings
        if sp_input_ids.shape[1] > bert_encoder_max_len - 1:
            # print(
            #     f'Text length is more than the maximum context length of the model ({bert_encoder_max_len}). Cutting from the BEGINNING of the text to max length.'
            # )
            sn_embed = sn_embed[:, : bert_encoder_max_len - 1]
            sn_mask = sn_mask[:, : bert_encoder_max_len - 1]

        logits, _ = self(
            sn_emd=sn_embed,
            sn_mask=sn_mask,
            word_ids_sn=word_ids_sn,
            sn_word_len=sn_word_len,
            sp_emd=sp_input_ids,
            sp_pos=sp_pos,
            sp_fix_dur=sp_fix_dur,
            sp_landing_pos=sp_landing_pos,
            word_ids_sp=word_ids_sp,
            scanpath=scanpath,
        )

        labels = batch_data.labels

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, labels)

        # Clear intermediate tensors to prevent memory buildup
        del scanpath, fixation_features, sp_input_ids, word_ids_sp
        del sp_pos, sp_fix_dur, sp_landing_pos, sn_embed, sn_mask
        del word_ids_sn, sn_word_len

        return labels, loss, logits.squeeze()


def pool_subword_to_word(
    subword_emb, word_ids_sn, target, max_seq_len, bert_dim, pool_method='sum'
):
    # batching computing
    # Pool bert token (subword) to word level
    if target == 'sn':
        max_len = max_seq_len  # CLS and SEP included
    elif target == 'sp':
        max_len = word_ids_sn.max().item() + 1  # +1 for the <s> token at the beginning
    else:
        raise NotImplementedError

    merged_word_emb = torch.empty(subword_emb.shape[0], 0, bert_dim).to(
        subword_emb.device
    )
    for word_idx in range(max_len):
        word_mask = (
            (word_ids_sn == word_idx)
            .unsqueeze(2)
            .repeat(1, 1, bert_dim)
            .to(subword_emb.device)
        )
        # pooling method -> sum
        if pool_method == 'sum':
            pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(
                1
            )  # [batch, 1, 1024]
        elif pool_method == 'mean':
            pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(
                1
            )  # [batch, 1, 1024]
        else:
            raise NotImplementedError
        merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

    mask_word = torch.sum(merged_word_emb, 2).bool()
    return merged_word_emb, mask_word


def pad_list(input_list, target_length, pad_with=0):
    # Calculate how many elements need to be added
    padding_length = target_length - len(input_list)

    # If padding_length is less than 0, the list is already longer than target_length
    if padding_length < 0:
        print('The list is already longer than the target length.')
        return input_list

    # Add padding_length number of zeros to the end of the list
    padded_list = input_list + [pad_with] * padding_length

    return padded_list


def get_word_length(word):
    if word in ['<s>', '</s>', '<pad>']:
        return 0
    else:
        return len(word.translate(str.maketrans('', '', string.punctuation)))


def align_word_ids_with_input_ids(
    tokenizer,
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: list,
):
    """
    Returns a tensor of the same shape as input_ids, containing the word index of each input_id (token)
    """
    word_ids_sn_lst = []
    retokenized_sn = tokenizer(
        decoded_to_txt_input_ids,
        return_tensors='pt',
    )
    for i in range(input_ids.shape[0]):
        word_ids_sn_lst.append(retokenized_sn.word_ids(i)[1:-1])

    word_ids_sn = torch.tensor(word_ids_sn_lst).to(input_ids.device)

    return word_ids_sn


def get_sn_word_lens(input_ids: torch.Tensor, decoded_to_txt_input_ids: list):
    def compute_p_lengths(p, target_length):
        return pad_list([get_word_length(word) for word in p], target_length)

    target_len = input_ids.shape[1]
    sn_word_len = torch.tensor(
        [
            compute_p_lengths(paragraph, target_len)
            for paragraph in decoded_to_txt_input_ids
        ]
    ).to(input_ids.device)

    return sn_word_len


def convert_positions_to_words_sp(
    scanpath: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    roberta_tokenizer_prefix_space,
):
    sp_tokens_strs = []
    for i in range(scanpath.shape[0]):
        curr_sp_tokens = [roberta_tokenizer_prefix_space.cls_token] + [
            decoded_to_txt_input_ids[i][word_i + 1]  # +1 to skip the <s> token
            for word_i in scanpath[i].tolist()
            if word_i != SCANPATH_PADDING_VAL
        ]
        curr_sp_tokens_str = ' '.join(curr_sp_tokens)
        sp_tokens_strs.append(curr_sp_tokens_str.split())

    return sp_tokens_strs


def eyettention_legacy_code(scanpath, fixation_features):
    # sp_pos is batch_data.scanpath, when adding 2 to each element that is not -1, add a 0 column at the beginning and add 1 to the wholte tensor
    sp_pos = scanpath.clone()
    sp_pos[sp_pos != SCANPATH_PADDING_VAL] += SCANPATH_PADDING_VAL
    sp_pos = torch.cat(
        (torch.zeros(sp_pos.shape[0], 1).to(sp_pos.device), sp_pos), dim=1
    )
    sp_pos += 1
    sp_pos = sp_pos.int()

    # unused_sp_ordinal_pos = batch_data.fixation_features[:, :, 0].int() #! TODO why not used? delete?

    sp_fix_dur = fixation_features[
        ..., 1
    ]  #! The feature order is hard coded in model_args. Make sure it's correct
    sp_landing_pos = fixation_features[..., 2]

    # add a column of zeros to both sp_fix_dur and sp_landing_pos to account for the <s> token
    sp_fix_dur = torch.cat(
        (torch.zeros(sp_fix_dur.shape[0], 1).to(sp_fix_dur.device), sp_fix_dur),
        dim=1,
    )
    sp_landing_pos = torch.cat(
        (
            torch.zeros(sp_landing_pos.shape[0], 1).to(sp_landing_pos.device),
            sp_landing_pos,
        ),
        dim=1,
    )

    return sp_pos, sp_fix_dur, sp_landing_pos


def calc_sp_word_input_ids(
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    roberta_tokenizer_prefix_space: object,
    scanpath: torch.Tensor,
):
    """This function calculates the word input ids for the scanpath

    Returns:
        word_ids_sp (torch.Tensor): The word index of each input_id (token) in the scanpath text
        sp_input_ids (torch.Tensor): The input ids of the scanpath text

    Args:
        input_ids (torch.Tensor): The word sequence input ids.
                Tensor of (batch_size, max_text_length_in_tokens)
        decoded_to_txt_input_ids (list): The decoded input ids.
                (list of lists of strings)
        roberta_tokenizer_prefix_space: The tokenizer with add_prefix_space=True
        scanpath (torch.Tensor): A scanpath tensor containing the word indices in the scanpath order
                Tensor of (batch_size, max_scanpath_length_in_words)
    """
    SP_word_ids, SP_input_ids = [], []

    sp_tokens_strs = convert_positions_to_words_sp(
        scanpath=scanpath,
        decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        roberta_tokenizer_prefix_space=roberta_tokenizer_prefix_space,
    )

    tokenized_SPs = roberta_tokenizer_prefix_space.batch_encode_plus(
        sp_tokens_strs,
        add_special_tokens=False,
        truncation=False,
        padding='longest',
        return_attention_mask=True,
        is_split_into_words=True,
    )
    for i in range(scanpath.shape[0]):
        encoded_sp = tokenized_SPs['input_ids'][i]
        word_ids_sp = tokenized_SPs.word_ids(i)  # -> Take the <sep> into account
        word_ids_sp = [
            val if val is not None else SCANPATH_PADDING_VAL for val in word_ids_sp
        ]

        SP_word_ids.append(word_ids_sp)
        SP_input_ids.append(encoded_sp)

    word_ids_sp = torch.tensor(SP_word_ids).to(input_ids.device)
    sp_input_ids = torch.tensor(SP_input_ids).to(input_ids.device)

    return word_ids_sp, sp_input_ids
