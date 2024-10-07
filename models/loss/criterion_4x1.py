"""
PET model
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)
import math


class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef  # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4: 0.5}

    def loss_labels(self, pair_data, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        # assert 'pred_logits' in outputs
        pred_logits, _, _, tgt_labels, match_point_weights = pair_data
        pred_logits = torch.cat(pred_logits, dim=0)

        src_logits = pred_logits
        idx = self._get_src_permutation_idx(indices)[1]
        tgt_labels = torch.cat(tgt_labels, dim=0)
        idx_tgt = torch.cat([J for  (_, J) in indices])
        target_classes_o = tgt_labels[idx_tgt] # torch.cat([t[J] for t, (_, J) in zip(tgt_labels, indices)])
        target_classes = torch.zeros(src_logits.shape[:1], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        assert target_classes.sum() == target_classes_o.sum()
        if src_logits.shape[0] == 0:
            loss_ce = 0.0
        else:
            loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, pair_data, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        # assert 'pred_points' in outputs
        _, pred_points, tgt_points, tgt_labels, match_point_weights = pair_data
        pred_points = torch.cat(pred_points, dim=0)
        # get indices
        idx = self._get_src_permutation_idx(indices)[1]
        src_points = pred_points[idx]
        # target_points = torch.cat(tgt_points, dim=0)
        idx_tgt = torch.cat([J for (_, J) in indices])
        target_points = torch.cat(tgt_points, dim=0)
        target_points = target_points[idx_tgt] # torch.cat([t[i] for t, (_, i) in zip(tgt_points, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs[0]['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        if src_points.shape[0] == 0:
            loss_points = 0.0
        else:
            loss_points = F.smooth_l1_loss(src_points, target_points, reduction='mean')
        losses['loss_points'] = loss_points
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, pair_data, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](pair_data, outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, seg_level_split_masks, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets

        pair_data_list = []
        img_H, img_W = outputs[0]['img_shape'] #targets[0]['seg_level_map'].shape[-2:]
        for lvl_idx, (batch_masks) in enumerate(seg_level_split_masks):

            batch_masks = F.interpolate(batch_masks.float().unsqueeze(1), size=outputs[0]['fea_shape']).squeeze(1).bool()
            filtered_pred_logits_list = []
            filtered_pred_points_list = []
            filtered_points_list = []
            filtered_labels_list = []
            filtered_match_point_weights_list = []


            # batch_masks 代表在同一个深度范围中的掩码
            for b_idx, (mask) in enumerate(batch_masks):

                pred_logits = outputs[lvl_idx]['pred_logits'][b_idx]
                pred_points = outputs[lvl_idx]['pred_points'][b_idx]
                mask_flatten = mask.flatten(0)

                filtered_pred_logits = pred_logits[mask_flatten]
                filtered_pred_points = pred_points[mask_flatten]
                filtered_pred_logits_list.append(filtered_pred_logits)
                filtered_pred_points_list.append(filtered_pred_points)


                points = targets[b_idx]['points']
                points = torch.clamp(points, min=0, max=img_H-1)

                # 1. 获取点的坐标
                y_coords = (points[:, 0] / 4).long()
                x_coords = (points[:, 1] / 4).long()

                valid_range = mask[y_coords, x_coords]
                assert valid_range.shape[0] == points.shape[0], f"{valid_range.shape[0]} 不等于 {  points.shape[0]}"
                filtered_pred_points = points[valid_range]

                labels = targets[b_idx]['labels']
                assert valid_range.shape[0] == labels.shape[0], f"{valid_range.shape[0]} 不等于 {  labels.shape[0]}"
                filtered_labels = labels[valid_range]

                match_point_weights = targets[b_idx]['match_point_weight']
                assert valid_range.shape[0] == match_point_weights.shape[-1], f"{valid_range.shape[0]} 不等于 {match_point_weights.shape[-1]}"
                filtered_match_point_weights = match_point_weights[..., valid_range]

                filtered_points_list.append(filtered_pred_points)
                filtered_labels_list.append(filtered_labels)
                filtered_match_point_weights_list.append(filtered_match_point_weights)

            pair_data_list.append((filtered_pred_logits_list, filtered_pred_points_list, filtered_points_list, filtered_labels_list, filtered_match_point_weights_list))

        indices_list = []
        for pair_data in pair_data_list:
            pred_logits, pred_points, tgt_points, tgt_labels, match_point_weights = pair_data
            indices = self.matcher((img_H, img_W), pred_logits, pred_points, tgt_points, tgt_labels, match_point_weights)
            indices_list.append(indices)

        num_points = -1
        # compute the average number of target points accross all nodes, for normalization purposes
        # num_points = sum(len(t["labels"]) for t in targets)
        # num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        # num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for lvl_idx, (indices, pair_data) in enumerate(zip(indices_list, pair_data_list)):
            for loss in self.losses:
                loss_val:dict = self.get_loss(loss, pair_data, outputs, targets, indices, num_points, **kwargs)
                for k in loss_val.keys():
                    if k in losses:
                        losses[k] += loss_val[k]
                    else:
                        losses[k] = loss_val[k]
        return losses, indices