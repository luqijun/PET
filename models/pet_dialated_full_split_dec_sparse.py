"""
PET model and criterion classes
"""

import torch.nn.functional as F

from util.misc import (check_and_clear_memory)
from .cc_head import CCHead
from .pet_base import PET_Base
from .transformer.dialated_prog_win_transformer_full_split_dec import build_encoder, build_decoder
from .utils import points_queris_embed


class PET(PET_Base):
    """
    Point quEry Transformer
    """

    def __init__(self, backbone, num_classes, args=None):
        super().__init__(backbone, num_classes, args)

        # predict head
        self.head_4x = CCHead(num_classes=1, num_pts_per_feature=4, args=args)
        self.head_8x = CCHead(num_classes=1, num_pts_per_feature=1, args=args)

    def get_build_enc_dec_func(self):
        return build_encoder, build_decoder

    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses

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

        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None

        encode_src = self.context_encoder(src, src_pos_embed, mask, **kwargs)
        context_info = (encode_src, src_pos_embed, mask)
        if 'test' in kwargs and clear_cuda_cache:
            check_and_clear_memory()

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_temp = F.interpolate(split_map, (sp_h, sp_w))
        split_map_dense = split_map_temp.reshape(bs, -1).unsqueeze(-1).repeat(1, 1, 4).reshape(bs, -1)
        split_map_sparse = 1 - split_map_temp.reshape(bs, -1)

        # quadtree layer0 forward (sparse)
        # level embeding
        kwargs['level_embed'] = self.level_embed[0]
        kwargs['dec_win_size_list'] = self.args.dec_win_size_list_8x  # [8, 4]
        kwargs['dec_win_dialation_list'] = self.args.dec_win_dialation_list_8x
        hs, points_queries = self.decode(self.transformer_decoder, samples,
                                         features['8x'], context_info, pq_stride=8, **kwargs)

        if 'train' in kwargs:
            outputs_sparse = self.head_8x(samples, points_queries, hs, pq_stride=8, **kwargs)
            outputs_dense = self.head_4x(samples, points_queries, hs, pq_stride=4, **kwargs)
            # outputs_sparse['fea_shape'] = features['8x'].tensors.shape[-2:]

        elif 'test' in kwargs:
            points_queries = points_queries.to(hs.device)

            sparse_mask = (split_map_sparse > 0.5).unsqueeze(0)
            sparse_hs = hs[sparse_mask].unsqueeze(0).unsqueeze(0)
            sparse_points_queries = points_queries.unsqueeze(0).unsqueeze(0)[sparse_mask]
            outputs_sparse = self.head_8x(samples, sparse_points_queries, sparse_hs, pq_stride=8, **kwargs)

            dense_mask = (split_map_sparse <= 0.5).unsqueeze(0)
            dense_hs = hs[dense_mask].unsqueeze(0).unsqueeze(0)
            dense_points_queries = points_queries.unsqueeze(0).unsqueeze(0)[dense_mask]
            outputs_dense = self.head_4x(samples, dense_points_queries, dense_hs, pq_stride=4, **kwargs)

        # format outputs
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs

    def decode(self, transformer, samples, features, context_info, pq_stride, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        src, _ = features.decompose()

        # get points queries for transformer
        # generate points queries and position embedding
        query_feats, query_embed, points_queries, qH, qW = \
            points_queris_embed(samples, pq_stride, src, **kwargs)

        pqs = (query_feats, query_embed, points_queries, qH, qW)

        # point querying
        kwargs['pq_stride'] = pq_stride
        kwargs['query_hw'] = (qH, qW)
        hs, points_queries = transformer(encode_src, src_pos_embed, mask, pqs,
                                         img_shape=samples.tensors.shape[-2:], **kwargs)
        return hs, points_queries


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
