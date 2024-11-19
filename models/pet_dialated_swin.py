"""
PET model and criterion classes
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from util.misc import (check_and_clear_memory)
from .layers import Segmentation_Head, MLP
from .pet_base import PET_Base
from .position_encoding import build_position_encoding
from .transformer.dialated_prog_swin_transformer import build_encoder
from .utils import points_queris


class PET(PET_Base):
    """
    Point quEry Transformer
    """

    def __init__(self, backbone, num_classes, args=None):
        nn.Module.__init__(self)
        self.args = args
        self.backbone = backbone

        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
        ]
        )

        # learn to scale
        self.learn_to_scale = args.get('learn_to_scale', False)
        if self.learn_to_scale:
            self.scale_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, padding=1),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.ReLU(),
            )

        # segmentation
        self.use_seg_head = args.get("use_seg_head", True)
        if self.use_seg_head:
            self.seg_head = Segmentation_Head(args.hidden_dim, 1)

        # quadtree splitter
        context_w = self.args.enc_win_size_list[-1][1]
        context_h = self.args.enc_win_size_list[-1][0]
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # context encoder
        self.encode_feats = '8x'
        self.enc_win_size_list = args.enc_win_size_list  # encoder window size
        self.enc_win_dialation_list = args.enc_win_dialation_list
        self.context_encoder = build_encoder(args, enc_win_size_list=self.enc_win_size_list,
                                             enc_win_dialation_list=self.enc_win_dialation_list)

        self.seg_level_split_th = args.seg_level_split_th
        self.warmup_ep = args.get("warmup_ep", 5)

        # level embeding
        self.level_embed = nn.Parameter(torch.Tensor(2, backbone.num_channels))
        normal_(self.level_embed)

        # loss
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.class_embed_sparse = nn.Linear(hidden_dim * 2, num_classes + 1)
        self.coord_embed_sparse = MLP(hidden_dim * 2, hidden_dim, 2, 3)

        self.class_embed_dense = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed_dense = MLP(hidden_dim, hidden_dim // 2, 2, 3)

    def get_build_enc_dec_func(self):
        return build_encoder, None

    def pet_forward(self, samples, features, pos, **kwargs):

        clear_cuda_cache = self.args.get("clear_cuda_cache", False)
        kwargs['clear_cuda_cache'] = clear_cuda_cache

        outputs = dict(seg_map=None, scale_map=None)
        fea_x8 = features['8x'].tensors

        # apply scale factors
        if self.learn_to_scale:
            scale_map = self.scale_head(fea_x8)
            pred_scale_map_4x = F.interpolate(scale_map, size=features['4x'].tensors.shape[-2:])
            pred_scale_map_8x = F.interpolate(scale_map, size=features['8x'].tensors.shape[-2:])
            features['4x'].tensors = features['4x'].tensors * pred_scale_map_4x
            features['8x'].tensors = features['8x'].tensors * pred_scale_map_8x
            outputs['scale_map'] = scale_map

        # apply seg head
        if self.use_seg_head:
            seg_map = self.seg_head(fea_x8)  # 已经经过sigmoid处理了
            pred_seg_map_4x = F.interpolate(seg_map, size=features['4x'].tensors.shape[-2:])
            pred_seg_map_8x = F.interpolate(seg_map, size=features['8x'].tensors.shape[-2:])
            features['4x'].tensors = features['4x'].tensors * pred_seg_map_4x
            features['8x'].tensors = features['8x'].tensors * pred_seg_map_8x
            outputs['seg_head_map'] = seg_map

        # apply quadtree splitter
        bs, _, src_h, src_w = fea_x8.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(fea_x8)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # context encoding
        src, mask = features['4x'].decompose()
        src_pos_embed = pos['4x']
        assert mask is not None

        encode_src = self.context_encoder(src, src_pos_embed, mask, **kwargs)
        context_info = (encode_src, src_pos_embed, mask)
        if 'test' in kwargs and clear_cuda_cache:
            check_and_clear_memory()

        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:

            points_queries = points_queris(samples, 8, encode_src[1], **kwargs).to(split_map_sparse.device)
            hs = encode_src[1].flatten(-2).transpose(1, 2).unsqueeze(0)
            if 'test' in kwargs:
                m = split_map_sparse.squeeze(0) > 0.5
                points_queries = points_queries[m]
                hs = hs[:, :, m, :]

            # level embeding
            kwargs['level'] = 'sparse'
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            outputs_sparse = self.predict(samples, points_queries, hs, **kwargs)
            outputs_sparse['fea_shape'] = features['8x'].tensors.shape[-2:]
            outputs_sparse['pq_stride'] = 8
        else:
            outputs_sparse = None

        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:

            points_queries = points_queris(samples, 4, encode_src[0], **kwargs).to(split_map_dense.device)
            hs = encode_src[0].flatten(-2).transpose(1, 2).unsqueeze(0)
            if 'test' in kwargs:
                m = split_map_dense.squeeze(0) > 0.5
                points_queries = points_queries[m]
                hs = hs[:, :, m, :]

            # level embeding
            kwargs['level'] = 'dense'
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            outputs_dense = self.predict(samples, points_queries, hs, **kwargs)
            outputs_dense['fea_shape'] = features['4x'].tensors.shape[-2:]
            outputs_dense['pq_stride'] = 4
        else:
            outputs_dense = None

        # format outputs
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs

    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """

        class_embed = self.class_embed_sparse if kwargs['level'] == 'sparse' else self.class_embed_dense
        coord_embed = self.coord_embed_sparse if kwargs['level'] == 'sparse' else self.class_embed_dense

        num_layers, bs, num_query, dim = hs.shape
        outputs_class = class_embed(hs)
        outputs_class = outputs_class.reshape(num_layers, bs, num_query, 2)
        # normalize to 0~1
        outputs_offsets = (coord_embed(hs).sigmoid() - 0.5) * 2.0
        outputs_offsets = outputs_offsets.reshape(num_layers, bs, num_query, 2)

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
        out = {
            'img_shape': img_shape,
            'pred_logits': outputs_class[-1],
            'pred_points': outputs_points,
            'pred_offsets': outputs_offsets[-1]
        }

        out['points_queries'] = points_queries
        return out


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
