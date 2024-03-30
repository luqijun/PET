"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import *

from .anchor_detr_layers import TransformerEncoderLayerSpatial
from .anchor_detr_layers import TransformerDecoderLayer
from ..utils import _get_clones, _get_activation_fn

class AnchorWinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False

        encoder_layer = TransformerEncoderLayerSpatial(d_model, dim_feedforward,
                                                       dropout, activation, nhead, "RCDA")
        self.encoder_layers = _get_clones(encoder_layer, num_encoder_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_pos_embed, mask, posemb_row=None, posemb_col=None, posemb_2d=None):
        bs, c, h, w = src.shape
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):

            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size

            posemb_row_tmp = posemb_row.unsqueeze(1).repeat(1, h, 1, 1).permute(0, 3, 1, 2)
            posemb_col_tmp = posemb_col.unsqueeze(2).repeat(1, 1, w, 1).permute(0, 3, 1, 2)

            memeory_win, src_pos_embed_win, posemb_row_tmp, posemb_col_tmp, mask_win = \
                enc_win_partition1(memeory, src_pos_embed, posemb_row_tmp, posemb_col_tmp, mask, enc_win_h, enc_win_w)

            use_src_pos = False
            if use_src_pos:
                posemb_row_tmp = (posemb_row_tmp + src_pos_embed_win).permute(0, 2, 3, 1)
                posemb_col_tmp = (posemb_col_tmp + src_pos_embed_win).permute(0, 2, 3, 1)
            else:
                posemb_row_tmp = posemb_row_tmp.permute(0, 2, 3, 1)
                posemb_col_tmp = posemb_col_tmp.permute(0, 2, 3, 1)

            output = self.encoder_layers[idx](memeory_win, mask_win, posemb_row_tmp, posemb_col_tmp,posemb_2d)

            # reverse encoder window
            memeory = enc_win_partition_reverse1(output, enc_win_h, enc_win_w, h, w)
            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class AnchorWinDecoderTransformer(nn.Module):
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
        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                dropout, activation, nhead,
                                                1, "RCDA")
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)
        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_shape, query_embed, points_queries, points_queries_origin, query_feats, v_idx, depth_embed = pqs

        win_partition_reverse_func = kwargs['win_partition_query_reverse_func']
        kwargs['v_idx'] = v_idx
        kwargs['query_shape'] = query_shape
        origin_h, origin_w = kwargs['img_shape']
        posemb_row = kwargs['posemb_row']
        posemb_col = kwargs['posemb_col']
        posemb_2d = kwargs['posemb_2d']
        adapt_pos1d = kwargs['adapt_pos1d']
        adapt_pos2d = kwargs['adapt_pos2d']


        tgt = query_feats
        query_pos = query_embed
        reference_points = points_queries_origin.float().to(tgt.device)
        reference_points[:, 0] = reference_points[:, 0] / origin_h
        reference_points[:, 1] = reference_points[:, 1] / origin_w
        srcs = src
        src_padding_masks = mask

        intermediates = []
        for idx in range(self.num_layer):
            tgt = self.decoder_layers[idx](tgt, query_pos, reference_points, srcs, src_pos_embed, src_padding_masks, **kwargs)
            if 'train' in kwargs:
                H, W = query_shape[-2:]
                tgt_unpartition = win_partition_reverse_func(tgt, H=H, W=W).permute(1, 0, 2)
            else:
                num_elm, num_win, dim = tgt.shape
                tgt_unpartition = tgt.reshape(1, num_elm*num_win, dim)
            intermediates.append(tgt_unpartition)

        return torch.stack(intermediates, dim=0)



def build_encoder(args, **kwargs):
    return AnchorWinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return AnchorWinDecoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
