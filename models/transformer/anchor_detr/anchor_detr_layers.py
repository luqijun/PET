# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .row_column_decoupled_attention import MultiheadRCDA
from ..utils import _get_clones, _get_activation_fn, mask2pos, pos2posemb1d, pos2posemb2d


class TransformerEncoderLayerSpatial(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # self attention
        self.self_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None,posemb_2d=None):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        if self.attention_type=="RCDA":

            src2 = self.self_attn((src + posemb_row).reshape(bz, h * w, c), (src + posemb_col).reshape(bz, h * w, c),
                                  src + posemb_row, src + posemb_col,
                                  src, key_padding_mask=padding_mask)[0].transpose(0, 1).reshape(bz, h, w, c)
        else:
            src2 = self.self_attn((src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  (src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  src.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=padding_mask.reshape(bz, h*w))[0].transpose(0, 1).reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0., activation="relu", n_heads=8,
                 n_levels=3, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # cross attention
        self.cross_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)


        # level combination
        if n_levels>1:
            self.level_fc = nn.Linear(d_model * n_levels, d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, srcs, srcs_pos, src_padding_masks=None, adapt_pos2d=None,
                adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None, v_idx=None, **kwargs):
        tgt_len = tgt.shape[1]

        # query_pos = adapt_pos2d(pos2posemb2d(reference_points))
        win_w, win_h = kwargs['dec_win_size_src']
        q_h, q_w = kwargs['query_shape'][-2:]
        win_partition_src_func = kwargs['win_partition_src_func']
        win_partition_query_func = kwargs['win_partition_query_func']
        # query_pos = query_pos.reshape(q_h, q_w, -1).unsqueeze(0)
        # query_pos = win_partition_func(query_pos)

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        bz, c, h, w = srcs.shape
        srcs = srcs.reshape(bz, c, h, w).permute(0, 2, 3, 1)
        srcs_pos = srcs_pos.reshape(bz, c, h, w).permute(0, 2, 3, 1)

        if self.attention_type == "RCDA":

            query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
            query_pos_x = query_pos_x.reshape(q_h, q_w, -1).unsqueeze(0).repeat(bz, 1, 1, 1).permute(0, 3, 1, 2)
            query_pos_x = win_partition_query_func(query_pos_x)
            query_pos_x = query_pos_x[:, v_idx]

            query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
            query_pos_y = query_pos_y.reshape(q_h, q_w, -1).unsqueeze(0).repeat(bz, 1, 1, 1).permute(0, 3, 1, 2)
            query_pos_y = win_partition_query_func(query_pos_y)
            query_pos_y = query_pos_y[:, v_idx]

            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_row = win_partition_src_func(posemb_row.permute(0, 3, 1, 2)).permute(1, 0, 2)
            posemb_row = posemb_row[v_idx]
            posemb_row = posemb_row.reshape(posemb_row.shape[0], win_h, win_w, -1)

            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            posemb_col = win_partition_src_func(posemb_col.permute(0, 3, 1, 2)).permute(1, 0, 2)
            posemb_col = posemb_col[v_idx]
            posemb_col = posemb_col.reshape(posemb_col.shape[0], win_h, win_w, -1)

            src_padding_masks = win_partition_src_func(src_padding_masks.unsqueeze(1)).permute(1, 0, 2).squeeze(-1)
            src_padding_masks = src_padding_masks[v_idx]
            src_padding_masks = src_padding_masks.reshape(src_padding_masks.shape[0], win_h, win_w)

            win_srcs = win_partition_src_func(srcs.permute(0, 3, 1, 2)).permute(1, 0, 2)
            win_srcs = win_srcs[v_idx]
            win_srcs = win_srcs.reshape(win_srcs.shape[0], win_h, win_w, -1)
            src_row = src_col = win_srcs

            use_src_pos = False
            if use_src_pos:
                win_srcs_pos = win_partition_src_func(srcs_pos.permute(0, 3, 1, 2)).permute(1, 0, 2)
                win_srcs_pos = win_srcs_pos[v_idx]
                win_srcs_pos = win_srcs_pos.reshape(win_srcs_pos.shape[0], win_h, win_w, -1)
                k_row = src_row + win_srcs_pos + posemb_row
                k_col = src_col + win_srcs_pos + posemb_col
            else:
                k_row = src_row + posemb_row
                k_col = src_col  + posemb_col

            tgt2 = self.cross_attn((tgt + query_pos_x).permute(1,0, 2), (tgt + query_pos_y).permute(1,0, 2),
                                   k_row, k_col,
                                   win_srcs, key_padding_mask=src_padding_masks)[0].transpose(0, 1)
        else:
            tgt2 = self.cross_attn((tgt + query_pos).transpose(0, 1),
                                   (srcs + posemb_2d).reshape(bz, h * w, c).transpose(0,1),
                                   srcs.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bz, h*w))[0].transpose(0,1)


        tgt2 = tgt2.permute(1, 0, 2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt


class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

