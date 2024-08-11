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
        self.split_depth_th = 0.4
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
        self.seg_head = Segmentation_Head(args.hidden_dim, 1)

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_dense = PETDecoder(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

        # bce loss
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
        out_dense = outputs['dense']

        out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
        valid_dense = out_dense_scores > thrs
        index_dense = valid_dense.cpu()

        # format output
        div_out = dict()
        output_names = out_dense.keys() if out_dense is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                div_out[name] = out_dense[name][index_dense].unsqueeze(0)
            else:
                div_out[name] = out_dense[name]

        gt_split_map = 1 - (torch.cat([tgt['depth'] for tgt in kwargs['targets']], dim=0) > self.split_depth_th).long()
        div_out['gt_split_map'] = gt_split_map
        div_out['gt_seg_head_map'] = torch.cat([tgt['seg_map'].unsqueeze(0) for tgt in kwargs['targets']], dim=0)
        div_out['pred_seg_head_map'] = F.interpolate(outputs['seg_map'], size=div_out['gt_seg_head_map'].shape[-2:]).squeeze(1)
        return div_out

    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses

    def pet_forward(self, samples, features, pos, **kwargs):

        fea_x8 = features['8x'].tensors
        # apply seg head
        seg_map = self.seg_head(fea_x8)
        seg_attention = seg_map.sigmoid()
        for fea in features.values():
            fea.tensors = fea.tensors * F.interpolate(seg_attention, size=fea.tensors.shape[-2:])

        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None

        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        kwargs['dec_win_size'] = [8, 4]
        outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        
        # format outputs
        outputs = dict(seg_map=None)
        outputs['dense'] = outputs_dense
        outputs['seg_map'] = seg_map
        return outputs

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_dense = outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        loss_dict_dense = criterion(output_dense, targets)

        # dense point queries loss
        loss_dict_dense = {k + '_ds': v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k + '_ds': v for k, v in weight_dict.items()}
        loss_pq_dense = sum(
            loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)

        # point queries loss
        losses = loss_pq_dense

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_dense)
        
        # seg head loss
        seg_map = outputs['seg_map']
        gt_seg_map = torch.stack([tgt['seg_map'] for tgt in targets], dim=0)
        gt_seg_map = F.interpolate(gt_seg_map.unsqueeze(1), size=seg_map.shape[-2:]).squeeze(1)
        loss_seg_map = self.bce_loss(seg_map.float().squeeze(1), gt_seg_map)
        losses += loss_seg_map * 0.1
        loss_dict['loss_seg_map'] = loss_seg_map * 0.1

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
