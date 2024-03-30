"""
PET model classes
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer.utils import mask2pos, pos2posemb1d, pos2posemb2d
from .transformer.anchor_detr.anchor_win_transformer import *
from .position_encoding import build_position_encoding
from .crowd_set_criterion import CrowdSetCriterion
import math
from .base_pet import BasePETCount
    

class CrowdPET(nn.Module):
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

        self.adapt_pos1d = nn.Sequential(
            nn.Linear(backbone.num_channels, backbone.num_channels),
            nn.ReLU(),
            nn.Linear(backbone.num_channels, backbone.num_channels),
        )
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(backbone.num_channels, backbone.num_channels),
            nn.ReLU(),
            nn.Linear(backbone.num_channels, backbone.num_channels),
        )

        # encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.encoder = build_encoder(args, enc_win_list=enc_win_list)

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
        self.sparse_stride = args.sparse_stride
        self.dense_stride = args.dense_stride

        # decoder 共享权重？
        transformer1 = build_decoder(args)
        transformer2 = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer1)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer2)
        # self._reset_parameters()

    # 加了之后效果很差
    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

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
        depth_embed = torch.cat([pos2posemb1d(tgt['depth']) for tgt in kwargs['targets']])
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

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()  # 8x特征
        src_pos_embed = pos[self.encode_feats] # 8x特征编码
        assert mask is not None

        # row col postion embedding
        pos_col, pos_row = mask2pos(mask)
        posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
        posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
        posemb_2d = None

        encode_src = self.encoder(src, src_pos_embed, mask, posemb_row, posemb_col, posemb_2d)
        context_info = (encode_src, src_pos_embed, mask)

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)

        # use prediction
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        split_map_raw = split_map

        # use depth
        # split_map = torch.stack([tgt['depth_level'] for tgt in kwargs['targets']], dim=0)
        # split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        # split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        # if 'train' in kwargs:
        #     split_map_raw = self.quadtree_splitter(encode_src)
        # else:
        #     split_map_raw = split_map

        # quadtree layer0 forward (sparse)
        kwargs.update(dict(
            posemb_row=posemb_row,
            posemb_col=posemb_col,
            posemb_2d=posemb_2d,
            adapt_pos1d=self.adapt_pos1d,
            adapt_pos2d=self.adapt_pos2d
        ))
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            kwargs['pq_stride'] = self.sparse_stride
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None

        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            kwargs['pq_stride'] = self.dense_stride
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
        loss_dict_sparse = {k + '_sp': v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k + '_sp': v for k, v in weight_dict.items()}
        loss_pq_sparse = sum(
            loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k + '_ds': v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k + '_ds': v for k, v in weight_dict.items()}
        loss_pq_dense = sum(
            loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)

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
        pred_depth_levels = outputs['split_map_raw']
        gt_depth_levels = []
        for tgt in targets:
            depth = F.adaptive_avg_pool2d(tgt['depth'], pred_depth_levels.shape[-2:])
            depth_level = torch.ones(depth.shape, device=depth.device)
            depth_level[depth > 0.4] = 0
            gt_depth_levels.append(depth_level)
        gt_depth_levels = torch.cat(gt_depth_levels, dim=0)
        # gt_depth_levels = torch.cat([target['depth_level'] for target in targets], dim=0)
        # pred_depth_levels = F.interpolate(outputs['split_map_raw'], size=gt_depth_levels.shape[-2:])
        loss_split_depth = F.binary_cross_entropy(pred_depth_levels.float().squeeze(1), gt_depth_levels)
        loss_split = loss_split_depth
        weight_split = 0.1  # if epoch >= warmup_ep else 0.0
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
        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}

def build_CrowdPET(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    backbone = build_backbone_vgg(args)
    model = CrowdPET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = CrowdSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion
