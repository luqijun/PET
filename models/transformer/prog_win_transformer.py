"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""

import torch

from .layers import TransformerEncoder, TransformerDecoder, EncoderLayer, DecoderLayer
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

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, mask, **kwargs):
        bs, c, h, w = src.shape

        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            memeory_win, pos_embed_win, mask_win = enc_win_partition(memeory, pos_embed, mask, enc_win_h, enc_win_w)

            # encoder forward
            output = self.encoder.single_forward(memeory_win, src_key_padding_mask=mask_win, pos=pos_embed_win,
                                                 layer_idx=idx, **kwargs)

            # reverse encoder window
            memeory = enc_win_partition_reverse(output, enc_win_h, enc_win_w, h, w)
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

        final_hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                                    query_pos=query_embed, **kwargs)

        # bs = tgt.shape[1]
        # group_num = 128  # 32 if tgt.shape[0]==32 else 128
        # if bs <= group_num or not kwargs['clear_cuda_cache']:
        #     final_hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
        #                                 query_pos=query_embed, **kwargs)
        # else:
        #     num_groups = bs // group_num
        #     num_left = bs % group_num
        #     group_sizes = [group_num] * num_groups
        #     if num_left > 0:
        #         group_sizes.append(num_left)
        #
        #     tgt_split = torch.split(tgt, group_sizes, dim=1)
        #     memory_win_split = torch.split(memory_win, group_sizes, dim=1)
        #     mask_win_split = torch.split(mask_win, group_sizes, dim=0)
        #     pos_embed_win_split = torch.split(pos_embed_win, group_sizes, dim=1)
        #     query_embed_split = torch.split(query_embed, group_sizes, dim=1)
        #     final_hs_win = []
        #     for one_tgt, one_memory_win, one_mask_win, one_pos_embed_win, one_query_embed in zip(tgt_split,
        #                                                                                          memory_win_split,
        #                                                                                          mask_win_split,
        #                                                                                          pos_embed_win_split,
        #                                                                                          query_embed_split):
        #         hs_win = self.decoder(one_tgt, one_memory_win, memory_key_padding_mask=one_mask_win,
        #                               pos=one_pos_embed_win,
        #                               query_pos=one_query_embed, **kwargs)
        #         final_hs_win.append(hs_win)
        #         torch.cuda.empty_cache()
        #     final_hs_win = torch.cat(final_hs_win, dim=2)

        num_layer, num_elm, num_win, dim = final_hs_win.shape
        hs = final_hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs

    def forward(self, src, pos_embed, mask, pqs, **kwargs):

        query_embed, points_queries, query_feats, v_idx, depth_embed = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']

        # window-rize memory input
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask,
                                                                int(self.dec_win_h / div_ratio),
                                                                int(self.dec_win_w / div_ratio))

        # dynamic decoder forward
        if 'test' in kwargs:
            memory_win = memory_win[:, v_idx]
            pos_embed_win = pos_embed_win[:, v_idx]
            mask_win = mask_win[v_idx]
            hs = self.decoder_forward_dynamic(query_feats, query_embed,
                                              memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w,
                                              src.shape,
                                              **kwargs)
            return hs
        else:
            # decoder forward
            hs = self.decoder_forward(query_feats, query_embed,
                                      memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape,
                                      **kwargs)
            return hs.transpose(1, 2)


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
