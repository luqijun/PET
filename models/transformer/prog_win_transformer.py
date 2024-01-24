"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

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

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, pos_embed, mask):
        bs, c, h, w = src.shape
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            memeory_win, pos_embed_win, mask_win  = enc_win_partition(memeory, pos_embed, mask, enc_win_h, enc_win_w)            

            # encoder forward
            output = self.encoder.single_forward(memeory_win, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx)

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
        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        query_embed_ = query_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                                                                        query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs
    
    def forward(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_embed, points_queries, query_feats, v_idx = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']
        
        # window-rize memory input
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask, 
                                                    int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        
        # dynamic decoder forward
        if 'test' in kwargs:
            memory_win = memory_win[:,v_idx]
            pos_embed_win = pos_embed_win[:,v_idx]
            mask_win = mask_win[v_idx]
            hs = self.decoder_forward_dynamic(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs
        else:
            # decoder forward
            hs = self.decoder_forward(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs.transpose(1, 2).contiguous()


    def forward_label_points(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']
        query_embeds, query_feats, points_queries, points_queries_offsets, masks = pqs
        # query_embed, points_queries, query_feats, v_idx = pqs

        # window-rize memory input
        stride = kwargs['pq_stride']
        div_ratio = 1 if stride == 8 else 2
        h_win = int(self.dec_win_h / div_ratio)
        w_win = int(self.dec_win_w / div_ratio)

        h_nums = (h // h_win)
        w_nums = (w // w_win)
        num_wins = h_nums * w_nums
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask, h_win, w_win)
        # mask_win:((bs*num_wins), win_size)
        # dynamic decoder forward
        if 'test' in kwargs:
            memory_win = memory_win[:, v_idx]
            pos_embed_win = pos_embed_win[:, v_idx]
            mask_win = mask_win[v_idx]
            hs = self.decoder_forward_dynamic(query_feats, query_embed,
                                              memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w,
                                              src.shape, **kwargs)
            return hs
        else:


            # new version
            q_win_list = []
            q_pos_list = []
            q_mask_list = []

            target_points = []

            window_index_mask = torch.zeros(src.shape[0] * num_wins, dtype=torch.bool)
            sq_len = 512 if stride == 8 else 128

            batch_indices = torch.nonzero(torch.Tensor([len(tgt['points']) for tgt in kwargs['targets']])).squeeze(1)
            origin_points_list = [tgt['points'] for tgt in kwargs['targets'] if len(tgt['points'])>0]

            for i, (query_embed, query_feat, point_query) in enumerate(zip(query_embeds, query_feats, points_queries)):

                batch_index = batch_indices[i]
                query_embed = query_embed.permute(1, 0)
                query_feat = query_feat.permute(1, 0)

                point_query = point_query // 8  # (n, 2)
                point_win_indexes = (point_query[:, 0] // h_win) * w_nums + point_query[:, 1] // w_win
                # assert torch.all(torch.logical_and(point_win_indexes >= 0, point_win_indexes < num_wins)), f"存在元素不在(0,{num_wins})范围内"

                origin_points = origin_points_list[i]
                for win_num in range(num_wins):
                    point_filter = point_win_indexes == win_num
                    q_w = query_feat[point_filter].unsqueeze(1)
                    q_p = query_embed[point_filter].unsqueeze(1)

                    if len(q_w)==0:
                        continue
                    else:
                        window_index_mask[batch_index * num_wins + win_num] = True

                    target_points.extend(origin_points[point_filter])

                    origin_len = q_w.shape[0]
                    pad_size = sq_len - origin_len
                    q_w = F.pad(q_w, (0, 0, 0, 0, 0, pad_size))
                    q_p = F.pad(q_p, (0, 0, 0, 0, 0, pad_size))

                    q_m = torch.ones(sq_len)
                    q_m[0:origin_len] = 0

                    q_win_list.append(q_w)
                    q_pos_list.append(q_p)
                    q_mask_list.append(q_m.bool())

            assert len(window_index_mask) <= memory_win.shape[1]
            memory_win = memory_win[:, window_index_mask]
            pos_embed_win = pos_embed_win[:, window_index_mask]
            mask_win = mask_win[window_index_mask]

            q_win = torch.cat(q_win_list, dim=1)
            q_pos = torch.cat(q_pos_list, dim=1)
            q_mask = torch.stack(q_mask_list, dim=0).bool().to(device=q_win.device)

            hs = self.decoder(q_win, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win,
                              query_pos=q_pos, tgt_key_padding_mask=q_mask, **kwargs)

            hs_result = []
            for i in range(q_mask.shape[0]):
                q_m = (~q_mask[i])
                h = hs[:,:,i:i+1]
                h = h[:, q_m]
                hs_result.append(h)

            hs_result = torch.cat([h for h in hs_result if h.shape[1]!=0], dim=1).transpose(1, 2).contiguous()
            target_points = torch.stack(target_points, dim=0)
            return hs_result, target_points

            # old version
            # memory_win = memory_win.reshape(h_win * w_win, bs, num_wins, c).permute(1, 0, 2, 3)
            # pos_embed_win = pos_embed_win.reshape(h_win * w_win, bs, num_wins, c).permute(1, 0, 2, 3)
            # mask_win =  mask_win.reshape(bs, num_wins, h_win * w_win)
            # hs_result = []
            #
            # for i, (query_embed, query_feat, point_query, memory_w, pos_w, mask_w) in enumerate(zip(query_embeds, query_feats, points_queries, memory_win, pos_embed_win, mask_win)):
            #     point_query = point_query // 8 # (n, 2)
            #     point_win_indexes = (point_query[:, 0] // h_win + 1) * (point_query[:, 1] // w_win + 1) - 1   # (n, )
            #     query_embed = query_embed.permute(1, 0)
            #     query_feat = query_feat.permute(1, 0)
            #
            #     hs_list = []
            #     if len(query_feat) > 0:
            #         for win_num in range(num_wins):
            #             point_filter = point_win_indexes == win_num
            #             q = query_feat[point_filter].unsqueeze(1)
            #             q_pos = query_embed[point_filter].unsqueeze(1)
            #
            #             if len(q) == 0:
            #                 continue
            #
            #             ori_mem = memory_w[:, win_num, :].unsqueeze(1)
            #             ori_pos = pos_w[:, win_num, :].unsqueeze(1)
            #             ori_mask = mask_w[win_num, :].unsqueeze(0)
            #
            #             hs = self.decoder(q, ori_mem, memory_key_padding_mask=ori_mask, pos=ori_pos,
            #                          query_pos=q_pos, **kwargs)
            #             hs_list.append(hs)
            #
            #     hs_list = torch.cat(hs_list, dim=1).cuda() if len(hs_list)>0 else torch.tensor([]).cuda()
            #     hs_result.append(hs_list)
            #     # assert hs_list.shape[1] == len(query_feat)
            #
            # return hs_result

        

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
