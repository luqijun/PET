import torch
import torch.nn.functional as F
from torch import nn
from .transformer import *
from functools import partial

class BasePETCount(nn.Module):
    """
    Base PET model
    """

    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        depth_input_embed = kwargs['depth_input_embed']  # B, C, H, W
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

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
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        # depth_embed
        depth_embed = depth_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = depth_embed.shape[:2]
        depth_embed = depth_embed.view(bs, c, h, w)

        return query_embed, points_queries, query_feats, depth_embed

    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        query_embed, points_queries, query_feats, depth_embed = self.points_queris_embed(samples, self.pq_stride,
                                                                                         src, **kwargs)

        query_shape = query_embed.shape
        win_partition_func = kwargs['win_partition_query_func']
        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = win_partition_func(query_embed) # win_h*win_w, B * num_wins, C
        depth_embed_win = win_partition_func(depth_embed)
        query_feats_win = win_partition_func(query_feats)



        if 'test' in kwargs:
            # dynamic point query generation
            div = kwargs['div']
            thrs = 0.5
            div_win = win_partition_func(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
            valid_div = (div_win > thrs).sum(dim=0)[:, 0]
            v_idx = valid_div > 0
            query_embed_win = query_embed_win[:, v_idx]
            query_feats_win = query_feats_win[:, v_idx]
            depth_embed_win = depth_embed_win[:, v_idx]

            h, w = query_embed.shape[-2:]
            points_queries_temp = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
            points_queries_win = win_partition_func(points_queries_temp)
            points_queries_win = points_queries_win.to(v_idx.device)
            points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)
        else:
            v_idx = torch.ones(query_embed_win.shape[1], device=query_embed_win.device).bool()
            points_queries_win = points_queries


        out = (query_shape, query_embed_win, points_queries_win, points_queries, query_feats_win, v_idx, depth_embed_win)
        return out

    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[..., 0] /= (img_h / 256)
            outputs_offsets[..., 1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape,
               'pred_offsets': outputs_offsets[-1]}

        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        new_dec_win_w = dec_win_w // div_ratio
        new_dec_win_h = dec_win_h// div_ratio
        kwargs['dec_win_size_src'] = [new_dec_win_w, new_dec_win_h]
        win_partition_query_func = partial(window_partition, window_size_h=dec_win_h, window_size_w=dec_win_w)
        win_partition_query_reverse_func = partial(window_partition_reverse, window_size_h=dec_win_h, window_size_w=dec_win_w)
        win_partition_src_func = partial(window_partition, window_size_h=new_dec_win_h, window_size_w=new_dec_win_w)
        kwargs['win_partition_query_func'] = win_partition_query_func
        kwargs['win_partition_query_reverse_func'] = win_partition_query_reverse_func
        kwargs['win_partition_src_func'] = win_partition_src_func

        # get points queries for transformer (query_embed, points_queries, query_feats, v_idx, depth_embed)
        pqs = self.get_point_query(samples, features, **kwargs)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        points_queries = pqs[2]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs



class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x