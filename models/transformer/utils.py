"""
Transformer window-rize tools
"""
import torch

def enc_win_partition1(src, pos_embed, posmb_row, posmb_col, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition1(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition1(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    posmb_row_win = window_partition1(posmb_row, window_size_h=enc_win_h, window_size_w=enc_win_w)
    posmb_col_win = window_partition1(posmb_col, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition1(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(1)
    return src_win, pos_embed_win, posmb_row_win, posmb_col_win, mask_win

def enc_win_partition_reverse1(inputs, window_size_h, window_size_w, H, W):
    """
    inputs: B * num_wins, win_h, win_w, C
    reverse window-rized input for encoder
    """
    num_wins = (H * W / window_size_h / window_size_w)
    B = int(inputs.shape[0] / num_wins)
    x = inputs.permute(0, 2, 3, 1).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x

def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(-1).permute(1,0)
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x


def window_partition1(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C) # (B * num_wins, win_h, win_w, C)
    windows = windows.permute(0,3,1,2) # B * num_wins, win_h, win_w, C
    return windows

def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C) # (B * num_wins, win_h, win_w, C)
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2) # win_h*win_w, B * num_wins, C
    return windows


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
