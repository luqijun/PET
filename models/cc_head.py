"""
PET Decoder model
"""
from torch import nn

from .layers import *
from .utils import expand_anchor_points


def zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


class CCHead(nn.Module):
    """ 
    Base PET model
    """

    def __init__(self, num_classes=1, num_pts_per_feature=1, args=None, **kwargs):
        super().__init__()
        self.num_pts_per_feature = num_pts_per_feature
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, (num_classes + 1) * self.num_pts_per_feature)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2 * self.num_pts_per_feature, 3)
        # self.coord_embed.apply(zero_init)

    def forward(self, samples, points_queries, hs, pq_stride, **kwargs):
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
            points_queries = expand_anchor_points(points_queries, pq_stride, with_origin=False)

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
        out['points_queries'] = points_queries
        out['pq_stride'] = pq_stride
        return out
