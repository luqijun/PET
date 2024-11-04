"""
Transformer window-rize tools
"""

import copy

import torch.nn as nn
import torch.nn.functional as F


def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(
        -1).permute(1, 0)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1, 0, 2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0, 3, 1, 2)
    return x


def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)

    # window_size*window_size, B*num_windows, C
    windows = windows.reshape(-1, window_size_h * window_size_w, C).permute(1, 0, 2)

    return windows


def window_unpartition(windows, window_size_h, window_size_w, H, W):
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1, 0, 2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
    return x


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1, 0, 2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H * W, -1).permute(1, 0, 2)
    return x


def win_partion_with_dialated(src, strideHW, win_sizes):
    B, C, H, W = src.shape
    strideH, strideW = strideHW

    # strideH*strideW*B, C, H, W
    newH, newW = H // strideH, W // strideW
    src = src.reshape(B, C, newH, strideH, newW, strideW) \
        .permute(3, 5, 0, 1, 2, 4).flatten(0, 2)

    win_h, win_w = win_sizes

    # window_size*window_size,strideH*strideW*B*num_windows, C
    src_win = window_partition(src, win_h, win_w).contiguous()

    return src_win, (newH, newW)


def win_unpartion_with_dialated(src_win, strideHW, win_sizes, newHW):
    win_h, win_w = win_sizes

    newH, newW = newHW
    strideH, strideW = strideHW

    # B*strideH*strideW, C, H, W
    src = window_unpartition(src_win, win_h, win_w, newH, newW)

    B = src.shape[0] // strideH // strideW
    oriH, oriW = newH * strideH, newW * strideW
    src = src.reshape(strideH, strideW, B, -1, newH, newW) \
        .permute(2, 3, 4, 0, 5, 1).reshape(B, -1, oriH, oriW).contiguous()
    return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
