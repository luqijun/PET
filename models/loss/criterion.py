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

    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes = torch.stack([idx[0] for idx in indices], dim=0)

        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort) // 2]
            sp_idx = den_sort[len(den_sort) // 2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights == 0] = self.empty_weight[0]
            weights[weights == 1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs

        target_points = torch.cat([idx[1] for idx in indices])
        src_points  = torch.cat([idx[2] for idx in indices])

        batch_idx = torch.cat([torch.full_like(src[3], i) for i, src in enumerate(indices)]).long()
        src_idx = torch.cat([idx[3] for idx in indices]).long()
        idx = batch_idx, src_idx

        # get indices
        # idx = self._get_src_permutation_idx(indices)
        # src_points = outputs['pred_points'][idx]
        # target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort) // 2]
            img_sp_idx = den_sort[len(den_sort) // 2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (
                        non_div_mask[idx].sum() + eps)

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / len(loss_points_raw)

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

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs, targets)
        indices = self.assign(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized() and num_points.device.type != 'cpu':
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


    def assign(self, outputs, targets):

        query_shape = outputs['query_shape']
        anchor_points = outputs['anchor_points']
        anchor_bboxes = outputs['anchor_bboxes']

        pred_points = outputs['pred_points']

        result = []
        for single_anchor_bboxes, single_anchor_points, single_pred_points, tgt in \
                zip(anchor_bboxes, anchor_points, pred_points, targets):
            single_gt_bboxes = tgt['gt_bboxes'] # (n, 4) ( y1, x1, y2, x2)
            single_gt_points = tgt['points'] # (n, 2) (y, x)
            single_gt_labels = tgt['labels'] # (n , 1)

            num_gt, num_priors = single_gt_points.size(0), single_anchor_bboxes.size(0)
            if num_gt == 0:
                assigned_labels = single_pred_points.new_full((num_priors,), 0 , dtype=torch.long)
                assigned_target_points = torch.empty((0, 2), device=single_pred_points.device)
                assigned_src_points = torch.empty((0, 2), device=single_pred_points.device)
                pos_inds = torch.empty((0,), device=single_pred_points.device)
                result.append((assigned_labels, assigned_target_points, assigned_src_points, pos_inds))
                continue

            overlaps = self.box_iou(single_anchor_bboxes, single_gt_bboxes)
            distances = torch.cdist(single_anchor_points.float(), single_gt_points, p=2)

            assert len(distances) > 0

            topk = 12
            selectable_k = min(topk, len(distances))
            _, candidate_idxs = distances.topk(selectable_k, dim=0, largest=False) # (topk, num_gt) 值代表行索引

            candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)] # (topk, num_gt)

            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :] # (topk, num_gt)

            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_priors

            priors_cx = single_anchor_points[:, 0]
            priors_cy = single_anchor_points[:, 1]
            ep_priors_cx = priors_cx.view(1, -1).expand(
                num_gt, num_priors).contiguous().view(-1) # (num_gt * num_priors)
            ep_priors_cy = priors_cy.view(1, -1).expand(
                num_gt, num_priors).contiguous().view(-1) # (num_gt * num_priors)
            candidate_idxs = candidate_idxs.view(-1) # (topk * num_gt)

            # calculate the left, top, right, bottom distance between positive
            # prior center and gt side
            l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - single_gt_bboxes[:, 0]
            t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - single_gt_bboxes[:, 1]
            r_ = single_gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
            b_ = single_gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts,
            # the one with the highest IoU will be selected.
            INF = 100000000
            overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1) # （num_gt * num_priors）
            index = candidate_idxs.view(-1)[is_pos.view(-1)] # 选择符合条件的候选点索引
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index] # 将overlaps值赋值给overlaps_inf
            overlaps_inf = overlaps_inf.view(num_gt, -1).t() # (num_priors, num_gt)

            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1) # (num_priors,)
            assigned_gt_inds = overlaps.new_full((num_priors,), 0, dtype=torch.long)
            assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1 # 代表每个预测点对应的gt的index

            # assigned_target_points = torch.zeros_like(single_anchor_points).float()
            # assigned_pred_points = torch.zeros_like(single_anchor_points).float()
            assigned_labels = assigned_gt_inds.new_full((num_priors,), 0)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()

            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = single_gt_labels[assigned_gt_inds[pos_inds] - 1]
                assigned_target_points = single_gt_points[assigned_gt_inds[pos_inds] - 1, :]
                assigned_src_points = single_pred_points[max_overlaps != -INF]  # 选择预测的点
                # assigned_target_points[pos_inds, :] = single_gt_points[assigned_gt_inds[pos_inds] - 1, :]
            else:
                assigned_target_points = torch.empty((0, 2), device=single_pred_points.device)
                assigned_src_points = torch.empty((0, 2), device=single_pred_points.device)

            if assigned_target_points.ndim == 1:
                assigned_target_points = assigned_target_points.unsqueeze(0)
                # assigned_src_points = assigned_src_points.unsqueeze(0)
                pos_inds = pos_inds.unsqueeze(0)

            result.append((assigned_labels, assigned_target_points, assigned_src_points, pos_inds))

        return result


    def box_iou(self, boxes1, boxes2):
        """
        Compute the intersection over union of two set of boxes.

        Args:
            boxes1 (Tensor[N, 4]): First set of boxes.
            boxes2 (Tensor[M, 4]): Second set of boxes.

        Returns:
            iou (Tensor[N, M]): Pairwise IoU between boxes from boxes1 and boxes2.
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou