"""
PET Decoder model
"""
from torch import nn

from .layers import *
from .utils import expand_anchor_points, points_queris_embed


def zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


class PETDecoder(nn.Module):
    """ 
    Base PET model
    """

    def __init__(self, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.num_pts_per_feature = args.get('num_pts_per_feature', 1)
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, (num_classes + 1) * self.num_pts_per_feature)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2 * self.num_pts_per_feature, 3)
        # self.coord_embed.apply(zero_init)

        # 预测头部大小
        self.use_pred_head_sizes = args.get('use_pred_head_sizes', False)
        if self.use_pred_head_sizes:
            self.sizes_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            self.sizes_embed.apply(zero_init)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

    def forward(self, transformer, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        src, _ = features[self.feat_name].decompose()

        # get points queries for transformer
        # generate points queries and position embedding
        query_feats, query_embed, points_queries, qH, qW = \
            points_queris_embed(samples, self.pq_stride, src, **kwargs)

        pqs = (query_feats, query_embed, points_queries, qH, qW)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        kwargs['query_hw'] = (qH, qW)
        hs, points_queries = transformer(encode_src, src_pos_embed, mask, pqs,
                                         img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs

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
        out = {
            'img_shape': img_shape,
            'pred_logits': outputs_class[-1],
            'pred_points': outputs_points,
            'pred_offsets': outputs_offsets[-1]
        }

        if self.use_pred_head_sizes:
            outputs_sizes = self.sizes_embed(hs)
            out['pred_sizes'] = outputs_sizes[-1]

        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out
