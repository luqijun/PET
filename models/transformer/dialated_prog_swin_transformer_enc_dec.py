"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .layers.swin_enc_dec_layers import TransformerEncoder, TransformerDecoder, EncoderLayer, DecoderLayer
from .utils import _get_clones, win_partion_with_dialated, win_unpartion_with_dialated, window_partition


def get_attn_mask(newH, newW, win_h, win_w, tgt_mask_windows=None, device=None):
    # img_mask = torch.zeros((strideH * strideW * B * 8, 1, newH, newW), device=memeory.device)  # 1 Hp Wp 1
    img_mask = torch.zeros((1, 1, newH, newW), device=device)  # 1 Hp Wp 1
    h_slices = (slice(0, -win_h),
                slice(-win_h, -win_h // 2),
                slice(-win_h // 2, None))
    w_slices = (slice(0, -win_w),
                slice(-win_w, -win_w // 2),
                slice(-win_w // 2, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, win_h, win_w).permute(1, 0,
                                                                    2)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, win_h * win_w)
    if tgt_mask_windows == None:
        tgt_mask_windows = mask_windows
    attn_mask = tgt_mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
        .masked_fill(attn_mask == 0, float(0.0)).bool()

    return attn_mask, mask_windows


class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """

    def __init__(self, d_model=256, win_size=(4, 8), nhead=8,
                 num_encoder_blocks=1,
                 num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, win_size, nhead, dim_feedforward, dropout, activation)

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

            B, C, H, W = memeory.shape
            strideH, strideW = stride, stride

            # 拆分 strideH*strideW*B, C, H, W
            newH, newW = H // strideH, W // strideW
            win_w, win_h = enc_win_size

            # 生成mask
            use_attn_mask = True
            if idx % 2 != 0 and use_attn_mask:
                attn_mask, _ = get_attn_mask(newH, newW, win_h, win_w, device=memeory.device)

                # roll
                shifted_memory = torch.roll(memeory, shifts=(-win_h // 2, -win_w // 2), dims=(2, 3))
                shifted_pos_embed = torch.roll(pos_embed, shifts=(-win_h // 2, -win_w // 2), dims=(2, 3))
                shifted_mask = torch.roll(mask.unsqueeze(1), shifts=(-win_h // 2, -win_w // 2), dims=(2, 3))

            else:
                shifted_memory = memeory
                shifted_pos_embed = pos_embed
                shifted_mask = mask.unsqueeze(1)
                attn_mask = None

            # encoder window partition
            memeory_win, newHW = win_partion_with_dialated(shifted_memory, (stride, stride), enc_win_size)
            pos_embed_win, _ = win_partion_with_dialated(shifted_pos_embed, (stride, stride), enc_win_size)
            mask_win, _ = win_partion_with_dialated(shifted_mask, (stride, stride), enc_win_size)
            mask_win = mask_win.squeeze(-1).permute(1, 0)

            # encoder forward
            kwargs['shift_size'] = (win_h // 2, win_w // 2)
            encoder_idx = 0 if len(self.encoders) == 1 else idx
            output = self.encoders[encoder_idx].single_forward(memeory_win,
                                                               src_mask=attn_mask,
                                                               src_key_padding_mask=mask_win,
                                                               pos=pos_embed_win, **kwargs)

            memeory = win_unpartion_with_dialated(output, (stride, stride), enc_win_size, newHW)

            if idx % 2 != 0 and use_attn_mask:
                # roll
                memeory = torch.roll(memeory, shifts=(win_h // 2, win_w // 2), dims=(2, 3))

            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class WinDecoderTransformer(nn.Module):
    """
    Transformer Decoder, featured with progressive rectangle window attention
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_blocks=1, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=False,
                 dec_win_w=16, dec_win_h=8,
                 ):
        super().__init__()
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)

        decoder_norm = nn.LayerNorm(d_model)
        self.num_decoder_blocks = num_decoder_blocks
        self.decoders = nn.Sequential(*[
            TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                               return_intermediate=return_intermediate_dec)
            for _ in range(num_decoder_blocks)
        ])
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
        return self.forward_new(src, pos_embed, mask, pqs, **kwargs)

    def forward_new(self, src, pos_embed, mask, pqs, **kwargs):

        query_feats, query_embed, points_queries, qH, qW = pqs

        is_train = 'train' in kwargs
        dec_win_size_list = kwargs['dec_win_size_list']
        dec_win_dialation_list = kwargs['dec_win_dialation_list']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2

        hs_intermediate_list = []
        for idx, (dec_win_size, stride) in enumerate(zip(dec_win_size_list, dec_win_dialation_list)):
            thrs = 0.5
            div = kwargs['div']

            B, C, H_tgt, W_tgt = query_feats.shape
            _, _, H_mem, W_mem = src.shape
            strideH, strideW = stride, stride

            # 拆分 strideH*strideW*B, C, H, W
            newH_tgt, newW_tgt = H_tgt // strideH, W_tgt // strideW
            newH_mem, newW_mem = H_mem // strideH, W_mem // strideW
            win_w, win_h = dec_win_size
            mem_dec_win_size = (dec_win_size[0] // div_ratio, dec_win_size[1] // div_ratio)
            win_w_mem, win_h_mem = mem_dec_win_size

            # 生成mask
            if idx % 2 != 0:

                tgt_attn_mask, tgt_mask_windows = get_attn_mask(newH_tgt, newW_tgt, win_h, win_w,
                                                                device=query_feats.device)
                mem_attn_mask, _ = get_attn_mask(newH_mem, newW_mem, win_h_mem, win_w_mem, tgt_mask_windows,
                                                 device=query_feats.device)

                # roll
                shifted_query_feats = torch.roll(query_feats, shifts=(-win_h // 2, -win_w // 2), dims=(2, 3))
                shifted_query_embed = torch.roll(query_embed, shifts=(-win_h // 2, -win_w // 2), dims=(2, 3))

                shifted_src = torch.roll(src, shifts=(-win_h_mem // 2, -win_w_mem // 2), dims=(2, 3))
                shifted_pos_embed = torch.roll(pos_embed, shifts=(-win_h_mem // 2, -win_w_mem // 2), dims=(2, 3))
                shifted_mask = torch.roll(mask.unsqueeze(1), shifts=(-win_h_mem // 2, -win_w_mem // 2), dims=(2, 3))

            else:
                shifted_query_feats = query_feats
                shifted_query_embed = query_embed
                tgt_attn_mask = None

                shifted_src = src
                shifted_pos_embed = pos_embed
                shifted_mask = mask.unsqueeze(1)
                mem_attn_mask = None

            # B, C, H, W = query_feats.shape
            tgt_win, qHW = win_partion_with_dialated(shifted_query_feats, (stride, stride), dec_win_size)
            tgt_pos_win, _ = win_partion_with_dialated(shifted_query_embed, (stride, stride), dec_win_size)

            mem_win, _ = win_partion_with_dialated(shifted_src, (stride, stride), mem_dec_win_size)
            mem_pos_win, _ = win_partion_with_dialated(shifted_pos_embed, (stride, stride), mem_dec_win_size)
            mem_mask_win, _ = win_partion_with_dialated(shifted_mask, (stride, stride), mem_dec_win_size)
            mem_mask_win = mem_mask_win.squeeze(-1).permute(1, 0).contiguous()

            # filter
            if not is_train:
                div_win, _ = win_partion_with_dialated(div.unsqueeze(1), (stride, stride), dec_win_size)

                valid_div = (div_win > thrs).sum(dim=0)[:, 0]
                v_idx = valid_div > 0

                tgt_win = tgt_win[:, v_idx]
                tgt_pos_win = tgt_pos_win[:, v_idx]
                mem_win = mem_win[:, v_idx]
                mem_pos_win = mem_pos_win[:, v_idx]
                mem_mask_win = mem_mask_win[v_idx, ...]

                if tgt_attn_mask != None:
                    tgt_attn_mask = tgt_attn_mask[v_idx, ...]
                if mem_attn_mask != None:
                    mem_attn_mask = mem_attn_mask[v_idx, ...]

                # 调整anchor_points
                if idx == len(dec_win_size_list) - 1:
                    points_queries = points_queries.reshape(qH, qW, 2).permute(2, 0, 1).unsqueeze(0)
                    points_queries_win, _ = win_partion_with_dialated(points_queries, (stride, stride), dec_win_size)
                    points_queries_win = points_queries_win.to(v_idx.device)
                    points_queries = points_queries_win[:, v_idx].reshape(-1, 2)

            decoder_idx = 0 if len(self.decoders) == 1 else idx
            hs_win = self.decoders[decoder_idx](tgt_win, mem_win,
                                                tgt_mask=tgt_attn_mask,
                                                memory_mask=mem_attn_mask,
                                                memory_key_padding_mask=mem_mask_win,
                                                pos=mem_pos_win,
                                                query_pos=tgt_pos_win, **kwargs)
            hs_win = hs_win[-1]
            if is_train:
                # B, C, H, W
                query_feats = win_unpartion_with_dialated(hs_win, (stride, stride), dec_win_size, qHW)
                hs = query_feats.flatten(-2).transpose(1, 2).unsqueeze(0)
            else:

                # update query_feats
                query_feats_win, qHW = win_partion_with_dialated(shifted_query_feats, (stride, stride), dec_win_size)
                query_feats_win[:, v_idx] = hs_win
                query_feats = win_unpartion_with_dialated(query_feats_win, (stride, stride), dec_win_size, qHW)

                num_elm, num_win, dim = hs_win.shape
                hs = hs_win.reshape(1, num_elm * num_win, dim).unsqueeze(0)

            if idx % 2 != 0:
                # roll
                query_feats = torch.roll(query_feats, shifts=(win_h // 2, win_w // 2), dims=(2, 3))

            hs_intermediate_list.append(hs)

        hs_res = hs_intermediate_list[-1]
        return hs_res, points_queries

    def forward_new_no_split(self, src, pos_embed, mask, pqs, **kwargs):

        query_feats, query_embed, points_queries, qH, qW = pqs

        is_train = 'train' in kwargs
        dec_win_size_list = kwargs['dec_win_size_list']
        dec_win_dialation_list = kwargs['dec_win_dialation_list']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2

        hs_intermediate_list = []
        for idx, (dec_win_size, stride) in enumerate(zip(dec_win_size_list, dec_win_dialation_list)):
            thrs = 0.5
            div = kwargs['div']

            # B, C, H, W = query_feats.shape
            tgt_win, qHW = win_partion_with_dialated(query_feats, (stride, stride), dec_win_size)
            tgt_pos_win, _ = win_partion_with_dialated(query_embed, (stride, stride), dec_win_size)

            mem_dec_win_size = (dec_win_size[0] // div_ratio, dec_win_size[1] // div_ratio)
            mem_win, _ = win_partion_with_dialated(src, (stride, stride), mem_dec_win_size)
            mem_pos_win, _ = win_partion_with_dialated(pos_embed, (stride, stride), mem_dec_win_size)
            mem_mask_win, _ = win_partion_with_dialated(mask.unsqueeze(1), (stride, stride), mem_dec_win_size)
            mem_mask_win = mem_mask_win.squeeze(-1).permute(1, 0).contiguous()

            decoder_idx = 0 if len(self.decoders) == 1 else idx
            hs_win = self.decoders[decoder_idx](tgt_win, mem_win, memory_key_padding_mask=mem_mask_win,
                                                pos=mem_pos_win, query_pos=tgt_pos_win, **kwargs)
            hs_win = hs_win[-1]

            query_feats = win_unpartion_with_dialated(hs_win, (stride, stride), dec_win_size, qHW)
            hs = query_feats.flatten(-2).transpose(1, 2).unsqueeze(0)

            if not is_train and idx == len(dec_win_size_list) - 1:
                div = div.flatten()
                valid_div = (div > thrs)
                hs = hs[:, :, valid_div, :]
                points_queries = points_queries.to(valid_div.device)
                points_queries = points_queries[valid_div, :]

            hs_intermediate_list.append(hs)

        hs_res = hs_intermediate_list[-1]
        return hs_res, points_queries


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
                       src_mask: Optional[Tensor] = None,
                       src_key_padding_mask: Optional[Tensor] = None,
                       pos: Optional[Tensor] = None,
                       layer_idx=0, **kwargs):

        output = src
        layer = self.layers[layer_idx]

        final_output = layer(output, src_mask=src_mask,
                             src_key_padding_mask=src_key_padding_mask, pos=pos)

        return final_output

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, **kwargs):

        intermediate = []
        output = src
        for idx, layer in enumerate(self.layers):

            if idx % 2 != 0:
                output = torch.roll()

            output = layer(output, src_mask=src_mask if idx % 2 != 0 else None,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        win_size=args.win_size,
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
        num_decoder_blocks=args.dec_blocks,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
