from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import _get_clones, _get_activation_fn


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
    def __init__(self, d_model, win_size, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = WindowAttention(d_model, win_size, nhead, attn_drop=dropout, proj_drop=dropout)

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

        self.self_attn = WindowAttention(d_model, None, nhead, attn_drop=dropout, proj_drop=dropout)
        self.multihead_attn = WindowAttention(d_model, None, nhead, attn_drop=dropout, proj_drop=dropout)

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
        tgt2 = self.multihead_attn(q=self.with_pos_embed(tgt, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # feed-forward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.window_size != None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, value, attn_mask=None, key_padding_mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            atten_mask: (0/-inf) atten_mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        B_, Nq, C = q.shape
        _, Nkv, _ = k.shape

        # # TODO: q、k单独加上位置编码
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = self.q_linear(q).reshape(B_, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(k).reshape(B_, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(value).reshape(B_, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, Nq, Nkv) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Nq, Nkv)

        if key_padding_mask is not None:
            key_padding_mask = torch.zeros_like(key_padding_mask.float()).masked_fill(key_padding_mask, float(-100.0))
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn = attn + key_padding_mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1, 0, 2)

        return x, attn
