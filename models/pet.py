"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding
import math


from .base_pet import BasePETCount
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
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

        # context encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

        # depth adapt
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(backbone.num_channels, backbone.num_channels),
            nn.ReLU(),
            nn.Linear(backbone.num_channels, backbone.num_channels),
        )

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # splitter depth loss
        gt_depth_levels = torch.cat([target['depth_level'] for target in targets], dim=0)
        pred_depth_levels = F.interpolate(outputs['split_map_raw'], size=gt_depth_levels.shape[-2:])
        loss_split_depth = F.binary_cross_entropy(pred_depth_levels.float().squeeze(1), gt_depth_levels)
        loss_split = loss_split_depth
        weight_split = 0.1 # if epoch >= warmup_ep else 0.0
        losses += loss_split * weight_split

        # # quadtree splitter loss
        # den = torch.tensor([target['density'] for target in targets])   # crowd density
        # bs = len(den)
        # ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        # ds_div = outputs['split_map_raw'][ds_idx]
        # sp_div = 1 - outputs['split_map_raw']
        #
        # # constrain sparse regions
        # loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()
        #
        # # constrain dense regions
        # if sum(ds_idx) > 0:
        #     ds_num = ds_div.shape[0]
        #     loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        # else:
        #     loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # update quadtree splitter loss            
        # loss_split = loss_split_sp + loss_split_ds
        # weight_split = 0.1 if epoch >= warmup_ep else 0.0
        # loss_dict['loss_split'] = loss_split
        # weight_dict['loss_split'] = weight_split

        # final loss
        # losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def pos2posemb1d(self, pos, num_pos_feats=256, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., None] / dim_t
        posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return posemb

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)


        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # depth embedding
        depth_embed = torch.cat([self.pos2posemb1d(tgt['depth']) for tgt in kwargs['targets']])
        #kwargs['depth_input_embed'] = depth_embed.permute(0, 3, 1, 2)
        kwargs['depth_input_embed'] = depth_embed.permute(0, 3, 1, 2) # self.adapt_pos1d(depth_embed).permute(0, 3, 1, 2)

        # dense_input_depth = torch.cat([tgt['depth'].unsqueeze(0) for tgt in kwargs['targets']])
        # dense_input_depth = nested_tensor_from_tensor_list(dense_input_depth.repeat(1, 3, 1, 1))
        # depth_features, _ = self.depth_backbone(dense_input_depth)
        # features['4x'].tensors += depth_features['4x'].tensors
        # features['8x'].tensors += depth_features['8x'].tensors

        # features = features + depth_features
        # kwargs['dense_input_embed'] = dense_input_embed + dense_input_depth
        # kwargs['dense_input_embed'] = dense_input_embed + (dense_input_depth - 0.5) * 2

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   
        return out

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)

        # split_map = torch.stack([tgt['depth_level'] for tgt in kwargs['targets']], dim=0)
        # split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        # split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        # if 'train' in kwargs:
        #     split_map_raw = self.quadtree_splitter(encode_src)
        # else:
        #     split_map_raw = split_map

        # old
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        split_map_raw = split_map
        
        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None
        
        # format outputs
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map_raw
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        thrs = 0.5  # inference threshold
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']

        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out


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
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4:0.5}
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
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
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        # 使用深度权重
        # depth_weights = torch.cat([v["depth_weight"] for v in targets], dim=1).permute(1, 0) * 25
        # depth_weights = depth_weights[:len(loss_points_raw)]
        # loss_points_raw *= depth_weights

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
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
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
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
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    backbone = build_backbone_vgg(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion
