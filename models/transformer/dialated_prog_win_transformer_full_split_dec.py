"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""

import torch

from .dialated_prog_win_transformer import WinEncoderTransformer
from .layers import TransformerDecoder, DecoderLayer
from .utils import *


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

        div = kwargs.get('div', None)
        is_train = 'train' in kwargs or 'div' not in kwargs
        dec_win_size_list = kwargs['dec_win_size_list']
        dec_win_dialation_list = kwargs['dec_win_dialation_list']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2

        hs_intermediate_list = []
        for idx, (dec_win_size, stride) in enumerate(zip(dec_win_size_list, dec_win_dialation_list)):
            thrs = 0.5

            # B, C, H, W = query_feats.shape
            tgt_win, qHW = win_partion_with_dialated(query_feats, (stride, stride), dec_win_size)
            tgt_pos_win, _ = win_partion_with_dialated(query_embed, (stride, stride), dec_win_size)

            mem_dec_win_size = (dec_win_size[0] // div_ratio, dec_win_size[1] // div_ratio)
            mem_win, _ = win_partion_with_dialated(src, (stride, stride), mem_dec_win_size)
            mem_pos_win, _ = win_partion_with_dialated(pos_embed, (stride, stride), mem_dec_win_size)
            mem_mask_win, _ = win_partion_with_dialated(mask.unsqueeze(1), (stride, stride), mem_dec_win_size)
            mem_mask_win = mem_mask_win.squeeze(-1).permute(1, 0).contiguous()

            if not is_train:
                div_win, _ = win_partion_with_dialated(div.unsqueeze(1), (stride, stride), dec_win_size)

                valid_div = (div_win > thrs).sum(dim=0)[:, 0]
                v_idx = valid_div > 0

                tgt_win = tgt_win[:, v_idx]
                tgt_pos_win = tgt_pos_win[:, v_idx]
                mem_win = mem_win[:, v_idx]
                mem_pos_win = mem_pos_win[:, v_idx]
                mem_mask_win = mem_mask_win[v_idx, ...]

                # 调整anchor_points
                if idx == len(dec_win_size_list) - 1:
                    points_queries = points_queries.reshape(qH, qW, 2).permute(2, 0, 1).unsqueeze(0)
                    points_queries_win, _ = win_partion_with_dialated(points_queries, (stride, stride), dec_win_size)
                    points_queries_win = points_queries_win.to(v_idx.device)
                    points_queries = points_queries_win[:, v_idx].reshape(-1, 2)

            decoder_idx = 0 if len(self.decoders) == 1 else idx
            hs_win = self.decoders[decoder_idx](tgt_win, mem_win, memory_key_padding_mask=mem_mask_win,
                                                pos=mem_pos_win, query_pos=tgt_pos_win, **kwargs)
            hs_win = hs_win[-1]

            if is_train:
                # B, C, H, W
                query_feats = win_unpartion_with_dialated(hs_win, (stride, stride), dec_win_size, qHW)
                hs = query_feats.flatten(-2).transpose(1, 2).unsqueeze(0)
            else:

                # update query_feats
                query_feats_win, qHW = win_partion_with_dialated(query_feats, (stride, stride), dec_win_size)
                query_feats_win[:, v_idx] = hs_win
                query_feats = win_unpartion_with_dialated(query_feats_win, (stride, stride), dec_win_size, qHW)

                num_elm, num_win, dim = hs_win.shape
                hs = hs_win.reshape(1, num_elm * num_win, dim).unsqueeze(0)

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

    def forward_old(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_feats, query_embed, points_queries, qH, qW = pqs

        dec_win_size_list = kwargs['dec_win_size_list']
        dec_win_dialation_list = kwargs['dec_win_dialation_list']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        is_test = 'test' in kwargs

        hs_result_list = []
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

                    if is_test and div is not None:
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

            decoder_idx = 0 if len(self.decoders) == 1 else idx
            hs_win = self.decoders[decoder_idx](tgt_win, mem_win, memory_key_padding_mask=mem_mask_win,
                                                pos=mem_pos_win, query_pos=tgt_pos_win, **kwargs)

            if is_test and div is not None:
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
                    hs = query_feats.flatten(-2).transpose(1, 2).unsqueeze(0)
                else:
                    num_layer, num_elm, num_win, dim = hs_win.shape
                    hs = hs_win.reshape(num_layer, num_elm * num_win, dim).unsqueeze(1)
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
                    query_feats_res = torch.zeros_like(query_feats)
                    for i in range(stride):
                        for j in range(stride):
                            query_feats_res[:, :, i::stride, j::stride] = hs_win_split_list_reverse[split_idx]
                            split_idx += 1
                    query_feats = query_feats_res
                    hs = query_feats.flatten(-2).transpose(1, 2).unsqueeze(0)
                else:
                    hs_tmp = [window_partition_reverse(hs_w, self.dec_win_h, self.dec_win_w, qH, qW) for hs_w in hs_win]
                    hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
                    hs = hs.transpose(1, 2)

            hs_result_list.append(hs)

        if len(v_idx_list) > 0:
            dec_win_w, dec_win_h = kwargs['dec_win_size_list'][-1]
            points_queries = points_queries.reshape(qH, qW, 2).permute(2, 0, 1).unsqueeze(0)
            points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
            points_queries_win = points_queries_win.to(query_feats.device)
            points_queries = points_queries_win[:, v_idx_list[-1]].reshape(-1, 2)

        hs_res = hs_result_list[-1]
        return hs_res, points_queries


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
        num_decoder_blocks=args.dec_blocks,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
