"""
tools
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


def expand_anchor_points(points, stride, with_origin=True):
    step = stride // 4
    offsets = torch.tensor([[-step, -step], [-step, step], [step, step], [step, -step]]).to(points.device)
    points = points.unsqueeze(1)
    new_points = points + offsets.unsqueeze(0)
    if with_origin:
        new_points = torch.cat([points, new_points], dim=1)
    return new_points.flatten(0, 1)


def points_queris_embed(samples, stride=8, src=None, **kwargs):
    """
    Generate point query embedding during training
    """
    # dense position encoding at every pixel location
    dense_input_embed = kwargs['dense_input_embed']

    if 'level_embed' in kwargs:
        level_embed = kwargs['level_embed'].view(1, -1, 1, 1)
        dense_input_embed = dense_input_embed + level_embed

    # get image shape
    input = samples.tensors
    image_shape = torch.tensor(input.shape[2:])
    shape = (image_shape + stride // 2 - 1) // stride

    # generate point queries
    shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
    shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1, 0)  # 2xN --> Nx2
    h, w = shift_x.shape

    # get point queries embedding
    query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
    bs, c = query_embed.shape[:2]

    # get point queries features, equivalent to nearest interpolation
    shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
    query_feats = src[:, :, shift_y_down, shift_x_down]

    query_embed = query_embed.view(bs, c, h, w)
    query_feats = query_feats.view(bs, c, h, w)

    return query_feats, query_embed, points_queries, h, w
