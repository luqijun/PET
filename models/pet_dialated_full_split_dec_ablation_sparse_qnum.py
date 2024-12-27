"""
PET model and criterion classes
"""

import torch.nn.functional as F

from util.misc import (check_and_clear_memory)
from .pet_base import PET_Base
from .transformer.dialated_prog_win_transformer_full_split_dec import build_encoder, build_decoder


class PET(PET_Base):
    """
    Point quEry Transformer
    """

    def __init__(self, backbone, num_classes, args=None):
        super().__init__(backbone, num_classes, args)

    def get_build_enc_dec_func(self):
        return build_encoder, build_decoder

    def pet_forward(self, samples, features, pos, **kwargs):

        clear_cuda_cache = self.args.get("clear_cuda_cache", False)
        kwargs['clear_cuda_cache'] = clear_cuda_cache

        outputs = dict(seg_map=None, scale_map=None)
        fea_x8 = features['8x'].tensors

        # 高层级替换为8x特征的插值
        features['4x'].tensors = F.interpolate(features['8x'].tensors, size=features['4x'].tensors.shape[-2:])

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
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            # level embeding
            kwargs['level_embed'] = self.level_embed[0]
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size_list'] = self.args.dec_win_size_list_8x  # [8, 4]
            kwargs['dec_win_dialation_list'] = self.args.dec_win_dialation_list_8x
            outputs_sparse = self.quadtree_sparse(self.transformer_decoder, samples, features, context_info, **kwargs)
            outputs_sparse['fea_shape'] = features['8x'].tensors.shape[-2:]
        else:
            outputs_sparse = None

        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            # level embeding
            kwargs['level_embed'] = self.level_embed[1]
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size_list'] = self.args.dec_win_size_list_8x  # [8, 4]
            kwargs['dec_win_dialation_list'] = self.args.dec_win_dialation_list_8x
            outputs_dense = self.quadtree_dense(self.transformer_decoder, samples, features, context_info, **kwargs)
            outputs_dense['fea_shape'] = features['4x'].tensors.shape[-2:]
        else:
            outputs_dense = None

        # format outputs
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
