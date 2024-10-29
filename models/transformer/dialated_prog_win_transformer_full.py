"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils import *


class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
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
                    # memeory_split_arr.append(mem)

            memeory_win = torch.cat(memeory_win_list, dim=1)
            pos_embed_win = torch.cat(pos_embed_win_list, dim=1)
            mask_win = torch.cat(mask_win_list, dim=0)

            # memeory_win, pos_embed_win, mask_win  = enc_win_partition(memeory, pos_embed, mask, enc_win_h, enc_win_w)

            # encoder forward
            output = self.encoder.single_forward(memeory_win, src_key_padding_mask=mask_win, pos=pos_embed_win,
                                                 layer_idx=idx, **kwargs)

            # reverse encoder window
            output_split_list = torch.split(output, output.shape[1] // (stride * stride), dim=1)
            output_split_list = [enc_win_partition_reverse(output_one, enc_win_h, enc_win_w, h // stride, w // stride)
                                 for output_one in output_split_list]

            split_idx = 0
            memeory_res = torch.full_like(memeory, fill_value=0.0)
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
        bs, c, h, w = src.shape
        query_embed, points_queries, query_feats = pqs

        qH, qW = query_feats.shape[-2:]

        dec_win_size_list = kwargs['dec_win_size_list']
        dec_win_dialation_list = kwargs['dec_win_dialation_list']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        is_test = 'test' in kwargs
        for idx, (dec_win_size, stride) in enumerate(zip(dec_win_size_list, dec_win_dialation_list)):

            v_idx_list = []
            self.dec_win_w, self.dec_win_h = dec_win_size

            thrs = 0.5
            div = kwargs['div']

            tgt_win_list = []
            tgt_pos_win_list = []

            mem_win_list = []
            mem_pos_win_list = []
            mem_mask_win_list = []
            for i in range(stride):
                for j in range(stride):
                    # split query
                    query_feats_one = query_feats[:, :, i::stride, j::stride]  # (B, C, H // 2, W // 2)
                    query_embed_one = query_embed[:, :, i::stride, j::stride]

                    tgt_win = window_partition(query_feats_one, window_size_h=self.dec_win_h,
                                               window_size_w=self.dec_win_w)
                    tgt_pos_win = window_partition(query_embed_one, window_size_h=self.dec_win_h,
                                                   window_size_w=self.dec_win_w)

                    # split memory
                    mem_one = src[:, :, i::stride, j::stride]
                    mem_pos_one = pos_embed[:, :, i::stride, j::stride]
                    mem_mask_one = mask[:, i::stride, j::stride]
                    memory_win, mem_pos_win, mem_mask_win = enc_win_partition(mem_one, mem_pos_one, mem_mask_one,
                                                                              int(self.dec_win_h / div_ratio),
                                                                              int(self.dec_win_w / div_ratio))

                    if is_test:
                        div_one = div[:, i::stride, j::stride]
                        div_win = window_partition(div_one.unsqueeze(1), window_size_h=self.dec_win_h,
                                                   window_size_w=self.dec_win_w)
                        valid_div = (div_win > thrs).sum(dim=0)[:, 0]
                        v_idx = valid_div > 0
                        v_idx_list.append(v_idx)

                        tgt_win = tgt_win[:, v_idx]
                        tgt_pos_win = tgt_pos_win[:, v_idx]
                        memory_win = memory_win[:, v_idx]
                        mem_pos_win = mem_pos_win[:, v_idx]
                        mem_mask_win = mem_mask_win[v_idx, ...]

                    tgt_win_list.append(tgt_win)
                    tgt_pos_win_list.append(tgt_pos_win)

                    mem_win_list.append(memory_win)
                    mem_pos_win_list.append(mem_pos_win)
                    mem_mask_win_list.append(mem_mask_win)

            tgt_win = torch.cat(tgt_win_list, dim=1)
            tgt_pos_win = torch.cat(tgt_pos_win_list, dim=1)

            mem_win = torch.cat(mem_win_list, dim=1)
            mem_pos_win = torch.cat(mem_pos_win_list, dim=1)
            mem_mask_win = torch.cat(mem_mask_win_list, dim=0)

            hs_win = self.decoder(tgt_win, mem_win, memory_key_padding_mask=mem_mask_win,
                                  pos=mem_pos_win, query_pos=tgt_pos_win, **kwargs)

            if is_test:
                if idx < len(dec_win_size_list) - 1:
                    hs_win = hs_win[-1]
                    split_sizes = [vi.sum().item() for vi in v_idx_list]
                    hs_win_split_list = torch.split(hs_win, split_sizes, dim=1)

                    split_idx = 0
                    query_feats_res = query_feats.clone()
                    for i in range(stride):
                        for j in range(stride):
                            query_feats_one = query_feats_res[:, :, i::stride, j::stride]
                            tgt_win = window_partition(query_feats_one, window_size_h=self.dec_win_h,
                                                       window_size_w=self.dec_win_w)

                            tgt_win[:, v_idx] = hs_win_split_list[split_idx]
                            tgt_win = enc_win_partition_reverse(tgt_win, self.dec_win_h, self.dec_win_w, qH // stride,
                                                                qW // stride)
                            query_feats_res[:, :, i::stride, j::stride] = tgt_win
                            split_idx += 1
                    query_feats = query_feats_res
                else:
                    num_layer, num_elm, num_win, dim = hs_win.shape
                    hs = hs_win.reshape(num_layer, num_elm * num_win, dim)

            else:
                if idx < len(dec_win_size_list) - 1:

                    hs_win = hs_win[-1]

                    # reverse encoder window
                    hs_win_split_list = torch.split(hs_win, hs_win.shape[1] // (stride * stride), dim=1)
                    hs_win_split_list_reverse = [
                        enc_win_partition_reverse(output_one, self.dec_win_h, self.dec_win_w, qH // stride,
                                                  qW // stride) for output_one
                        in hs_win_split_list]

                    split_idx = 0
                    query_feats_res = torch.full_like(query_feats, fill_value=0.0)
                    for i in range(stride):
                        for j in range(stride):
                            query_feats_res[:, :, i::stride, j::stride] = hs_win_split_list_reverse[split_idx]
                            split_idx += 1
                    query_feats = query_feats_res
                else:
                    hs_tmp = [window_partition_reverse(hs_w, self.dec_win_h, self.dec_win_w, qH, qW) for hs_w in hs_win]
                    hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
                    hs = hs.transpose(1, 2)

        return hs, v_idx_list

    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w,
                        src_shape,
                        **kwargs):
        """
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        query_embed_ = query_embed.permute(1, 2, 0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                              query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs

    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h,
                                dec_win_w, src_shape,
                                **kwargs):
        """
        decoder forward during inference
        """
        # decoder attention
        tgt = query_feats

        bs = tgt.shape[1]
        group_num = 128  # 32 if tgt.shape[0]==32 else 128
        if bs <= group_num or not kwargs['clear_cuda_cache']:
            final_hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                                        query_pos=query_embed, **kwargs)
        else:
            num_groups = bs // group_num
            num_left = bs % group_num
            group_sizes = [group_num] * num_groups
            if num_left > 0:
                group_sizes.append(num_left)

            tgt_split = torch.split(tgt, group_sizes, dim=1)
            memory_win_split = torch.split(memory_win, group_sizes, dim=1)
            mask_win_split = torch.split(mask_win, group_sizes, dim=0)
            pos_embed_win_split = torch.split(pos_embed_win, group_sizes, dim=1)
            query_embed_split = torch.split(query_embed, group_sizes, dim=1)
            final_hs_win = []
            for one_tgt, one_memory_win, one_mask_win, one_pos_embed_win, one_query_embed in zip(tgt_split,
                                                                                                 memory_win_split,
                                                                                                 mask_win_split,
                                                                                                 pos_embed_win_split,
                                                                                                 query_embed_split):
                hs_win = self.decoder(one_tgt, one_memory_win, memory_key_padding_mask=one_mask_win,
                                      pos=one_pos_embed_win,
                                      query_pos=one_query_embed, **kwargs)
                final_hs_win.append(hs_win)
                torch.cuda.empty_cache()
            final_hs_win = torch.cat(final_hs_win, dim=2)

        num_layer, num_elm, num_win, dim = final_hs_win.shape
        hs = final_hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs


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

        bs = output.shape[1]
        group_num = 8  # if output.shape[0] == 512 else 128
        if 'train' in kwargs or bs <= group_num or not kwargs['clear_cuda_cache']:
            final_output = layer(output, src_mask=mask,
                                 src_key_padding_mask=src_key_padding_mask, pos=pos)
        else:
            # 分开计算
            num_groups = bs // group_num
            num_left = bs % group_num
            group_sizes = [group_num] * num_groups
            if num_left > 0:
                group_sizes.append(num_left)

            output_split = torch.split(output, group_sizes, dim=1)
            src_key_padding_mask_split = torch.split(src_key_padding_mask, group_sizes, dim=0)
            pos_split = torch.split(pos, group_sizes, dim=1)
            final_output = []
            for one_output, one_src_key_padding_mask, one_pos_split in zip(output_split, src_key_padding_mask_split,
                                                                           pos_split):
                one_output = layer(one_output, src_mask=None,
                                   src_key_padding_mask=one_src_key_padding_mask, pos=one_pos_split)
                final_output.append(one_output)
                torch.cuda.empty_cache()
            final_output = torch.cat(final_output, dim=1)

        return final_output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

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


class TransformerDecoder(nn.Module):
    """
    Base Transformer Decoder
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src = self.norm1(src)

        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                ):
        # decoder self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # decoder cross attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed-forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
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


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
