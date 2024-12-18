"""
PET model and criterion classes
"""

import torch
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

    def test_forward(self, samples, features, pos, **kwargs):
        thrs = 0.5  # inference threshold
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_sparse = outputs['sparse']

        # process sparse point queries
        out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
        valid_sparse = out_sparse_scores > thrs
        index_sparse = valid_sparse.cpu()

        # format output
        div_out = dict()
        output_names = out_sparse.keys()
        for name in list(output_names):
            if 'pred' in name:
                div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
            else:
                div_out[name] = out_sparse[name]

        # gt_split_map = 1 - (torch.cat([tgt['seg_level_map'] for tgt in kwargs['targets']],
        #                               dim=0) > self.seg_level_split_th).long()
        # div_out['gt_split_map'] = gt_split_map
        # div_out['pred_split_map'] = F.interpolate(outputs['split_map_raw'], size=gt_split_map.shape[-2:]).squeeze(1)
        # div_out['gt_seg_head_map'] = torch.cat([tgt['seg_head_map'].unsqueeze(0) for tgt in kwargs['targets']], dim=0)
        # div_out['pred_seg_head_map'] = F.interpolate(outputs['seg_head_map'], size=div_out['gt_seg_head_map'].shape[-2:]).squeeze(1)
        return div_out

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

        # quadtree layer0 forward (sparse)
        # level embeding
        kwargs['level_embed'] = self.level_embed[0]
        kwargs['dec_win_size_list'] = self.args.dec_win_size_list_8x  # [8, 4]
        kwargs['dec_win_dialation_list'] = self.args.dec_win_dialation_list_8x
        outputs_sparse = self.quadtree_sparse(self.transformer_decoder, samples, features, context_info, **kwargs)
        outputs_sparse['fea_shape'] = features['8x'].tensors.shape[-2:]

        # format outputs
        outputs['sparse'] = outputs_sparse
        return outputs

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse = outputs['sparse']
        weight_dict = criterion.weight_dict
        warmup_ep = self.warmup_ep

        # compute loss
        loss_dict_sparse, _ = criterion(output_sparse, targets)

        # sparse point queries loss
        loss_dict_sparse = {k + '_sp': v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k + '_sp': v for k, v in weight_dict.items()}
        loss_pq_sparse = sum(
            loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # point queries loss
        losses = loss_pq_sparse

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)

        # scale map loss
        if self.learn_to_scale:
            pred_scale_map = outputs['scale_map']
            scale_map_h, scale_map_w = pred_scale_map.shape[-2:]
            scale_values_list = []
            for batch_idx, tgt in enumerate(targets):
                pts = (tgt['points'] // 8).long()
                pts = torch.clamp(pts, max=scale_map_h - 1, min=0)
                scale_values = pred_scale_map[batch_idx, :, pts[:, 0], pts[:, 1]]
                scale_values_list.append(scale_values.squeeze(0))
            pred_scale_values = torch.cat(scale_values_list, dim=0)
            tgt_scale_values = 1 / torch.cat([tgt['head_sizes'] for tgt in targets], dim=0)

            loss_scale_values = self.l1_loss(pred_scale_values, tgt_scale_values)
            losses += loss_scale_values * 0.1
            loss_dict['loss_scale_values'] = loss_scale_values * 0.1

            # gt_scale_map = torch.stack([tgt['seg_level_map'] for tgt in targets], dim=0)
            # # resize seg map
            # if gt_scale_map.shape[-1] < pred_scale_map.shape[-1]:
            #     gt_scale_map = F.interpolate(gt_scale_map, size=pred_scale_map.shape[-2:])
            # else:
            #     pred_scale_map = F.interpolate(pred_scale_map, size=gt_scale_map.shape[-2:])
            #
            # loss_scale_map = self.l1_loss(pred_scale_map.float().squeeze(1), gt_scale_map.float().squeeze(1))
            # losses += loss_scale_map * 0.1
            # loss_dict['loss_scale_map'] = loss_scale_map * 0.1

        # seg head loss
        if self.use_seg_head:
            pred_seg_map = outputs['seg_head_map']
            gt_seg_map = torch.stack([tgt['seg_head_map'] for tgt in targets], dim=0)

            # resize seg map
            if gt_seg_map.shape[-1] < pred_seg_map.shape[-1]:
                gt_seg_map = F.interpolate(gt_seg_map, size=pred_seg_map.shape[-2:])
            else:
                pred_seg_map = F.interpolate(pred_seg_map, size=gt_seg_map.shape[-2:])
            # pred_seg_map = F.interpolate(pred_seg_map, size=gt_seg_map.shape[-2:])
            loss_seg_map = self.bce_loss(pred_seg_map.float().squeeze(1), gt_seg_map.float().squeeze(1))
            losses += loss_seg_map * 0.1
            loss_dict['loss_seg_head_map'] = loss_seg_map * 0.1

        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
