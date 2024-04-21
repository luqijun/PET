"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .loss import build_criterion
from .pet_decoder import PETDecoder
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding
from .utils import pos2posemb1d
from .layers import Segmentation_Head
    

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
        
        # segmentation
        # self.seg_head = Segmentation_Head(args.hidden_dim, 1)

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
        self.quadtree_sparse = PETDecoder(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = PETDecoder(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

        # depth adapt
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(backbone.num_channels, backbone.num_channels),
            nn.ReLU(),
            nn.Linear(backbone.num_channels, backbone.num_channels),
        )
        self.split_depth_th = 0.4
        
        # level embeding
        # self.level_embed = nn.Parameter(
        #     torch.Tensor(2, backbone.num_channels))
        # normal_(self.level_embed)

        self.bce_loss = nn.BCEWithLogitsLoss()

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
        kwargs['depth_input_embed'] = self.adapt_pos1d(depth_embed).permute(0, 3, 1, 2)

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   
        return out

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
                    div_out[name] = torch.cat(
                        [out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]

        gt_split_map = 1 - (torch.cat([tgt['depth'] for tgt in kwargs['targets']], dim=0) > self.split_depth_th).long()
        div_out['gt_split_map'] = gt_split_map
        div_out['gt_seg_head_map'] = torch.cat([tgt['seg_map'].unsqueeze(0) for tgt in kwargs['targets']], dim=0)
        if outputs['split_map_raw'] is not None:
            div_out['pred_split_map'] = F.interpolate(outputs['split_map_raw'], size=gt_split_map.shape[-2:]).squeeze(1)
        if outputs['seg_map'] is not None:
            div_out['pred_seg_head_map'] = F.interpolate(outputs['seg_map'], size=div_out['gt_seg_head_map'].shape[-2:]).squeeze(1)
        return div_out

    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)
        
        # apply seg head
        seg_map = None # self.seg_head(encode_src)
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # kwargs['div'] = None # split_map_sparse.reshape(bs, sp_h, sp_w)
        # kwargs['dec_win_size'] = [16, 8]
        # outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        # outputs_dense = None

        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            # level embeding
            # kwargs['level_embed'] = self.level_embed[0]
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            # level embeding
            # kwargs['level_embed'] = self.level_embed[1]
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None
        
        # format outputs
        outputs = dict(seg_map=None)
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        outputs['seg_map'] = seg_map
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
        # loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        # weight_dict.update(weight_dict_dense)
        
        # seg head loss
        if 'seg_map' in outputs and outputs['seg_map'] is not None:
            seg_map = outputs['seg_map']
            gt_seg_map = torch.stack([tgt['seg_map'] for tgt in targets], dim=0)
            gt_seg_map = F.interpolate(gt_seg_map.unsqueeze(1), size=seg_map.shape[-2:]).squeeze(1)
            loss_seg_map = self.bce_loss(seg_map.float().squeeze(1), gt_seg_map)
            losses += loss_seg_map * 0.1
            loss_dict['loss_seg_map'] = loss_seg_map

        # splitter depth loss
        pred_depth_levels = outputs['split_map_raw']
        gt_depth_levels = []
        for tgt in targets:
            depth = F.adaptive_avg_pool2d(tgt['depth'], pred_depth_levels.shape[-2:])
            depth_level = torch.ones(depth.shape, device=depth.device)
            depth_level[depth > self.split_depth_th] = 0
            gt_depth_levels.append(depth_level)
        gt_depth_levels = torch.cat(gt_depth_levels, dim=0)
        # gt_depth_levels = torch.cat([target['depth_level'] for target in targets], dim=0)
        # pred_depth_levels = F.interpolate(outputs['split_map_raw'], size=gt_depth_levels.shape[-2:])
        loss_split_depth = F.binary_cross_entropy(pred_depth_levels.float().squeeze(1), gt_depth_levels)
        loss_split = loss_split_depth
        losses += loss_split * 0.1
        loss_dict['loss_split_depth'] = loss_split

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


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    args.num_classes = num_classes
    backbone = build_backbone_vgg(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    criterion = build_criterion(args)
    criterion.to(device)
    return model, criterion
