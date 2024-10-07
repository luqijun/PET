"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .pet_decoder_uniform_4x1 import PETDecoder
from .backbones import *
from .transformer.dialated_prog_win_transformer_full_split_dec import build_encoder, build_decoder
from .position_encoding import build_position_encoding
from .utils import pos2posemb1d
from .layers import Segmentation_Head
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        self.seg_level_split_th_list = args.seg_level_split_th_list
        self.warmup_ep = args.get("warmup_ep", 5)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # context encoder
        self.encode_feats = '8x'
        self.enc_win_size_list = args.enc_win_size_list  # encoder window size
        self.enc_win_dialation_list = args.enc_win_dialation_list
        args.enc_layers = len(self.enc_win_size_list)
        self.context_encoder = build_encoder(args, enc_win_size_list=self.enc_win_size_list, enc_win_dialation_list=self.enc_win_dialation_list)
        
        # segmentation
        self.use_seg_head = args.get("use_seg_head", True)
        if self.use_seg_head:
            self.seg_head = Segmentation_Head(args.hidden_dim, 1)

        # quadtree splitter
        context_patch = torch.tensor(self.enc_win_size_list[-1]) * 8
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, len(self.seg_level_split_th_list) + 1, 1),
            nn.Sigmoid(),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_dense = PETDecoder(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

        # depth adapt
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(backbone.num_channels, backbone.num_channels),
            nn.ReLU(),
            nn.Linear(backbone.num_channels, backbone.num_channels),
        )

        
        # level embeding
        self.level_embed = nn.Parameter(
            torch.Tensor(2, backbone.num_channels))

        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        normal_(self.level_embed)

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
        depth_embed = torch.cat([pos2posemb1d(tgt['seg_level_map']) for tgt in kwargs['targets']])
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

        pred_logits = outputs['split_map_logits']
        pred_logits = F.interpolate(pred_logits, size=out_dense[0]['fea_shape'])
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)

        # 遍历每个类别
        masks = []
        C = pred_probs.shape[1]
        for class_idx in range(C):
            # 创建一个与 pred_classes 形状相同的布尔张量，表示当前类别的 mask
            mask = (pred_classes == class_idx)  # 形状为 (B, H, W)
            masks.append(mask.flatten(-2))

        # format output
        div_out = dict()
        output_names = out_dense[0].keys()
        for idx, (outputs_one) in enumerate(out_dense):
            # process dense point queries
            out_scores = torch.nn.functional.softmax(outputs_one['pred_logits'], -1)[..., 1]
            valid_mask = (out_scores > thrs) & masks[idx]
            valid_index = valid_mask.cpu()

            for name in list(output_names):
                if 'pred' in name:
                    if name in div_out:
                        div_out[name] = torch.cat([div_out[name], outputs_one[name][valid_index].unsqueeze(0)], dim=1)
                    else:
                        div_out[name] = outputs_one[name][valid_index].unsqueeze(0)
                else:
                    div_out[name] = outputs_one[name]

        # gt_split_map = 1 - (torch.cat([tgt['seg_level_map'] for tgt in kwargs['targets']], dim=0) > self.seg_level_split_th).long()
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

        outputs = dict(seg_map=None)
        fea_x8 = features['8x'].tensors

        # apply seg head
        if self.args.use_seg_head:
            seg_map = self.seg_head(fea_x8)  # 已经经过sigmoid处理了
            outputs['seg_head_map'] = seg_map
        if self.args.use_seg_head_attention:
            seg_map_attention = seg_map.sigmoid()
            pred_seg_map_4x = F.interpolate(seg_map_attention, size=features['4x'].tensors.shape[-2:])
            features['4x'].tensors = features['4x'].tensors * pred_seg_map_4x


        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None

        encode_src = self.context_encoder(src, src_pos_embed, mask, **kwargs)
        context_info = (encode_src, src_pos_embed, mask)

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map_logits = self.quadtree_splitter(encode_src)
        kwargs['split_map_logits'] = split_map_logits
        # split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        # split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # if self.args.use_seg_level_attention:
        #     dense_feats_shape = features['4x'].tensors.shape[-2:]
        #     seg_level_attention_dense_4x = split_map_dense.sigmoid().reshape(bs, 1, dense_feats_shape[0], dense_feats_shape[1])
        #     features['4x'].tensors = features['4x'].tensors * seg_level_attention_dense_4x

        # level embeding
        kwargs['level_embed'] = self.level_embed[1]
        kwargs['div'] = None # split_map_dense.reshape(bs, ds_h, ds_w)

        kwargs['dec_win_size_list'] = self.args.dec_win_size_list_4x # // 2 # [4, 2]
        kwargs['dec_win_dialation_list'] = self.args.dec_win_dialation_list_4x
        outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)

        
        # format outputs
        outputs['sparse'] = None
        outputs['dense'] = outputs_dense
        outputs['split_map_logits'] = split_map_logits
        # outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = None
        return outputs

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = self.warmup_ep

        pred_seg_levels = outputs['split_map_logits']
        gt_seg_level_maps = torch.cat([tgt['seg_level_map'] for tgt in targets], dim=0)
        gt_seg_level_maps = F.adaptive_avg_pool2d(gt_seg_level_maps, pred_seg_levels.shape[-2:])
        gt_seg_levels = torch.ones_like(gt_seg_level_maps, device=gt_seg_level_maps.device)
        split_th_start_list = [0] + self.seg_level_split_th_list
        split_th_end_list = self.seg_level_split_th_list + [1]

        seg_level_split_masks = []
        for idx, (th_start, th_end) in enumerate(zip(split_th_start_list, split_th_end_list)):
            mask = (gt_seg_level_maps > th_start) & (gt_seg_level_maps <= th_end)
            gt_seg_levels[mask] = idx
            seg_level_split_masks.append(mask)

        temp_split_th_tuple_list = []
        temp_seg_level_split_th_list = [0] + self.seg_level_split_th_list + [1]
        num_part = len(self.seg_level_split_th_list) + 1
        for idx in range(num_part):
            start_idx = max(idx-1, 0)
            end_idx = min(idx + 2, num_part)
            temp_split_th_tuple_list.append((temp_seg_level_split_th_list[start_idx], temp_seg_level_split_th_list[end_idx]))

        temp_seg_level_split_masks = []
        for idx, (th_start, th_end) in enumerate(temp_split_th_tuple_list):
            mask = (gt_seg_level_maps > th_start) & (gt_seg_level_maps <= th_end)
            gt_seg_levels[mask] = idx
            temp_seg_level_split_masks.append(mask)

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_dense, _ = criterion(output_dense, targets, temp_seg_level_split_masks, div=outputs['split_map_dense'])
        else:
            loss_dict_dense, _ = criterion(output_dense, targets, seg_level_split_masks)

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
            losses += loss_seg_map * self.args.seg_head_loss_weight
            loss_dict['loss_seg_head_map'] = loss_seg_map * self.args.seg_head_loss_weight

        # splitter depth loss
        loss_split_seg_level = self.ce_loss(pred_seg_levels.float(), gt_seg_levels.long())
        loss_split = loss_split_seg_level
        losses += loss_split * self.args.seg_level_loss_weight
        loss_dict['loss_seg_level_map'] = loss_split * self.args.seg_level_loss_weight

        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
