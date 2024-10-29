"""
PET Decoder model
"""
import torch
from torch import nn

from .layers import *
from .transformer import *
from .utils import expand_anchor_points


class PETDecoder(nn.Module):
    """ 
    Base PET model
    """

    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        self.num_pts_per_feature = args.get('num_pts_per_feature', 1)
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, (num_classes + 1) * self.num_pts_per_feature)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2 * self.num_pts_per_feature, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        src, _ = features[self.feat_name].decompose()

        # get points queries for transformer
        # generate points queries and position embedding
        query_feats_win, query_embed_win, points_queries, qH, qW = \
            self.points_queris_embed(samples, self.pq_stride, src, **kwargs)

        pqs = (query_feats_win, query_embed_win, points_queries, qH, qW)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        kwargs['query_hw'] = (qH, qW)
        hs, points_queries = self.transformer(encode_src, src_pos_embed, mask, pqs,
                                              img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs

    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
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

        # 拆分成window
        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        return query_feats_win, query_embed_win, points_queries, h, w

    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        num_layers, bs, num_query, dim = hs.shape
        outputs_class = self.class_embed(hs)
        outputs_class = outputs_class.reshape(num_layers, bs, num_query * self.num_pts_per_feature, 2)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0
        outputs_offsets = outputs_offsets.reshape(num_layers, bs, num_query * self.num_pts_per_feature, 2)

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape

        if self.num_pts_per_feature == 4:
            points_queries = expand_anchor_points(points_queries, self.pq_stride, with_origin=False)

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
