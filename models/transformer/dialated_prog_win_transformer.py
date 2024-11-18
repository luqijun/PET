"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TransformerEncoder, TransformerDecoder, EncoderLayer, DecoderLayer
from .utils import _get_clones, win_partion_with_dialated, win_unpartion_with_dialated, window_partition, \
    window_partition_reverse, enc_win_partition, enc_win_partition_reverse


class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """

    def __init__(self, d_model=256, nhead=8,
                 num_encoder_blocks=1,
                 num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        self.encoders = nn.Sequential(*[
            TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
            for _ in range(num_encoder_blocks)
        ])
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_size_list = kwargs['enc_win_size_list']
        self.enc_win_dialation_list = kwargs['enc_win_dialation_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, mask, **kwargs):
        return self.forward_new(src, pos_embed, mask, **kwargs)

    def forward_new(self, src, pos_embed, mask, **kwargs):

        memeory_list = []
        memeory = src  # (B, C, H, W)
        enc_win_dialation_list = self.enc_win_dialation_list
        for idx, enc_win_size in enumerate(self.enc_win_size_list):

            stride = enc_win_dialation_list[idx]

            # encoder window partition
            memeory_win, newHW = win_partion_with_dialated(memeory, (stride, stride), enc_win_size)
            pos_embed_win, _ = win_partion_with_dialated(pos_embed, (stride, stride), enc_win_size)
            mask_win, _ = win_partion_with_dialated(mask.unsqueeze(1), (stride, stride), enc_win_size)
            mask_win = mask_win.squeeze(-1).permute(1, 0)

            # encoder forward
            encoder_idx = 0 if len(self.encoders) == 1 else idx
            output = self.encoders[encoder_idx](memeory_win, src_key_padding_mask=mask_win,
                                                pos=pos_embed_win, **kwargs)

            memeory = win_unpartion_with_dialated(output, (stride, stride), enc_win_size, newHW)
            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_

    def forward_old(self, src, pos_embed, mask, **kwargs):
        bs, c, h, w = src.shape

        memeory_list = []
        memeory = src  # (B, C, H, W)
        enc_win_dialation_list = self.enc_win_dialation_list
        for idx, enc_win_size in enumerate(self.enc_win_size_list):
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size

            stride = enc_win_dialation_list[idx]
            # memeory_split_arr = []
            memeory_win_list = []
            pos_embed_win_list = []
            mask_win_list = []
            for i in range(stride):
                for j in range(stride):
                    mem_one = memeory[:, :, i::stride, j::stride]  # (B, C, H // 2, W // 2)
                    pos_one = pos_embed[:, :, i::stride, j::stride]
                    mask_one = mask[:, i::stride, j::stride]
                    memeory_win_one, pos_embed_win_one, mask_win_one = enc_win_partition(mem_one, pos_one, mask_one,
                                                                                         enc_win_h, enc_win_w)
                    memeory_win_list.append(memeory_win_one)
                    pos_embed_win_list.append(pos_embed_win_one)
                    mask_win_list.append(mask_win_one)

            memeory_win = torch.cat(memeory_win_list, dim=1)
            pos_embed_win = torch.cat(pos_embed_win_list, dim=1)
            mask_win = torch.cat(mask_win_list, dim=0)

            # encoder forward
            encoder_idx = 0 if len(self.encoders) == 1 else idx
            output = self.encoders[encoder_idx](memeory_win, src_key_padding_mask=mask_win,
                                                pos=pos_embed_win, **kwargs)

            # reverse encoder window
            output_split_list = torch.split(output, output.shape[1] // (stride * stride), dim=1)
            output_split_list = [enc_win_partition_reverse(output_one, enc_win_h, enc_win_w, h // stride, w // stride)
                                 for output_one in output_split_list]

            split_idx = 0
            memeory_res = torch.zeros_like(memeory)
            for i in range(stride):
                for j in range(stride):
                    memeory_res[:, :, i::stride, j::stride] = output_split_list[split_idx]
                    split_idx += 1
            memeory = memeory_res
            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class WinDecoderTransformer(nn.Module):
    """
    Transformer Decoder, featured with progressive rectangle window attention
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=False,
                 dec_win_w=16, dec_win_h=8,
                 ):
        super().__init__()
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.cross_attn_depth = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, mask, pqs, **kwargs):

        dec_win_w, dec_win_h = kwargs['dec_win_size_list'][-1]
        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        query_feats, query_embed, points_queries, qH, qW = pqs

        # 拆分成window
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # window-rize memory input
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask,
                                                                int(self.dec_win_h / div_ratio),
                                                                int(self.dec_win_w / div_ratio))

        # dynamic decoder forward
        if 'train' in kwargs:
            # decoder attention
            # 重点：训练阶段shape保持一致 有助于损失函数分层计算
            tgt = query_feats_win
            hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                                  query_pos=query_embed_win, **kwargs)
            hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
            hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])  # num_layers * HW * B * C

            return hs.transpose(1, 2), points_queries  # num_layers * B * HW * C

        else:

            points_queries = points_queries.reshape(qH, qW, 2).permute(2, 0, 1).unsqueeze(0)
            points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)

            thrs = 0.5
            div = kwargs['div']
            div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
            valid_div = (div_win > thrs).sum(dim=0)[:, 0]
            v_idx = valid_div > 0
            query_embed_win = query_embed_win[:, v_idx]
            query_feats_win = query_feats_win[:, v_idx]
            points_queries_win = points_queries_win.to(v_idx.device)
            points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)

            memory_win = memory_win[:, v_idx]
            pos_embed_win = pos_embed_win[:, v_idx]
            mask_win = mask_win[v_idx]

            # decoder attention
            tgt = query_feats_win
            final_hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                                        query_pos=query_embed_win, **kwargs)

            num_layer, num_elm, num_win, dim = final_hs_win.shape
            hs = final_hs_win.reshape(num_layer, num_elm * num_win, dim)
            return hs.unsqueeze(1), points_queries_win  # num_layers * 1 * （num_elm * num_win） * C


class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder
    """

    def __init__(self, encoder_layer, num_layers, **kwargs):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

        if 'return_intermediate' in kwargs:
            self.return_intermediate = kwargs['return_intermediate']
        else:
            self.return_intermediate = False

    def single_forward(self, src,
                       mask: Optional[Tensor] = None,
                       src_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None,
                       layer_idx=0, **kwargs):

        output = src
        layer = self.layers[layer_idx]

        final_output = layer(output, src_mask=mask,
                             src_key_padding_mask=src_key_padding_mask, pos=pos)

        # bs = output.shape[1]
        # group_num = 8  # if output.shape[0] == 512 else 128
        # if 'train' in kwargs or bs <= group_num or not kwargs['clear_cuda_cache']:
        #     final_output = layer(output, src_mask=mask,
        #                          src_key_padding_mask=src_key_padding_mask, pos=pos)
        # else:
        #     # 分开计算
        #     num_groups = bs // group_num
        #     num_left = bs % group_num
        #     group_sizes = [group_num] * num_groups
        #     if num_left > 0:
        #         group_sizes.append(num_left)
        #
        #     output_split = torch.split(output, group_sizes, dim=1)
        #     src_key_padding_mask_split = torch.split(src_key_padding_mask, group_sizes, dim=0)
        #     pos_split = torch.split(pos, group_sizes, dim=1)
        #     final_output = []
        #     for one_output, one_src_key_padding_mask, one_pos_split in zip(output_split, src_key_padding_mask_split,
        #                                                                    pos_split):
        #         one_output = layer(one_output, src_mask=None,
        #                            src_key_padding_mask=one_src_key_padding_mask, pos=one_pos_split)
        #         final_output.append(one_output)
        #         torch.cuda.empty_cache()
        #     final_output = torch.cat(final_output, dim=1)

        return final_output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, **kwargs):

        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_blocks=args.enc_blocks,
        num_encoder_layers=args.enc_layers,
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return WinDecoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
