"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import *

from .crowd_transformer import TransformerEncoderLayerSpatial
from .crowd_transformer import TransformerDecoderLayer

class CrowdWinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayerSpatial(d_model, dim_feedforward,
                                                       dropout, activation, nhead, "RCDA")

        self.encoder_layers = _get_clones(encoder_layer, num_encoder_layers)

        # encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
        #                                             dropout, activation)
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, pos_embed, mask, posemb_row=None, posemb_col=None,posemb_2d=None):
        bs, c, h, w = src.shape
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):

            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size

            posemb_row_tmp = posemb_row.unsqueeze(1).repeat(1, h, 1, 1).permute(0, 3, 1, 2)
            posemb_col_tmp = posemb_col.unsqueeze(2).repeat(1, 1, w, 1).permute(0, 3, 1, 2)

            memeory_win, pos_embed_win, posemb_row_tmp, posemb_col_tmp, mask_win = enc_win_partition1(memeory, pos_embed,
                                                                                              posemb_row_tmp, posemb_col_tmp, mask,
                                                                                              enc_win_h, enc_win_w)

            posemb_row_tmp = posemb_row_tmp.permute(0, 2, 3, 1)
            posemb_col_tmp = posemb_col_tmp.permute(0, 2, 3, 1)

            output = self.encoder_layers[idx](memeory_win, mask_win, posemb_row_tmp, posemb_col_tmp,posemb_2d)

            # reverse encoder window
            memeory = enc_win_partition_reverse1(output, enc_win_h, enc_win_w, h, w)
            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class CrowdWinDecoderTransformer(nn.Module):
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

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                dropout, activation, nhead,
                                                1, "RCDA")
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)


        # decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                     return_intermediate=return_intermediate_dec)
        # self.cross_attn_depth = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape,
                        depth_embed, **kwargs):
        """ 
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        query_embed_ = query_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        depth_embed_ = depth_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        depth_embed_win = window_partition(depth_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # depth attention
        # tgt += depth_embed_win
        # tgt = self.cross_attn_depth(tgt + query_embed_win, depth_embed_win + query_embed_win, depth_embed_win)[0]

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape,
                                depth_embed, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats

        # depth attention
        # tgt += depth_embed
        # tgt = self.cross_attn_depth(tgt + query_embed, depth_embed + query_embed, depth_embed)[0]

        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs
    
    def forward(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_shape, query_embed, points_queries, points_queries_origin, query_feats, v_idx, depth_embed = pqs

        # src = src.
        # query_embed = query_embed.flatten(2).permute(2, 0, 1)  # NxCxHxW --> (HW)xNxC
        # depth_embed = depth_embed.flatten(2).permute(2, 0, 1)
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
            tgt = self.decoder_layers[idx](tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs)
            if 'train' in kwargs:
                H, W = query_shape[-2:]
                tgt_unpartition = win_partition_reverse_func(tgt, H=H, W=W).permute(1, 0, 2)
            else:
                num_elm, num_win, dim = tgt.shape
                tgt_unpartition = tgt.reshape(1, num_elm*num_win, dim)
            intermediates.append(tgt_unpartition)

        return torch.stack(intermediates, dim=0)
        
        # # window-rize memory input
        # div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        # memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask,
        #                                             int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        #
        # # dynamic decoder forward
        # if 'test' in kwargs:
        #     memory_win = memory_win[:,v_idx]
        #     pos_embed_win = pos_embed_win[:,v_idx]
        #     mask_win = mask_win[v_idx]
        #     hs = self.decoder_forward_dynamic(query_feats, query_embed,
        #                                       memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape,
        #                                       depth_embed, **kwargs)
        #     return hs
        # else:
        #     # decoder forward
        #     hs = self.decoder_forward(query_feats, query_embed,
        #                               memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape,
        #                               depth_embed,  **kwargs)
        #     return hs.transpose(1, 2)
        

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
                layer_idx=0):
        
        output = src
        layer = self.layers[layer_idx]
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)        
        return output

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
    return CrowdWinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return CrowdWinDecoderTransformer(
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
