"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_
from .layers import *

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

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
        self.args = args
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
        self.context_encoder_4x = build_encoder(args, enc_win_list=enc_win_list)
        self.context_encoder_8x = build_encoder(args, enc_win_list=enc_win_list)

        # segmentation
        self.use_seg_head = args.get("use_seg_head", True)
        if self.use_seg_head:
            self.seg_head = Segmentation_Head(args.hidden_dim, 1)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, 3, 1),
        )

        self.seg_level_split_th = args.seg_level_split_th
        self.seg_level_split_range = args.seg_level_split_range
        self.warmup_ep = args.get("warmup_ep", 5)

        self.class_embed_4x = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed_4x = MLP(hidden_dim, hidden_dim, 2, 3)
        self.head_size_4x = MLP(hidden_dim, hidden_dim, 1, 3)

        self.class_embed_8x = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed_8x = MLP(hidden_dim, hidden_dim, 2, 3)
        self.head_size_8x = MLP(hidden_dim, hidden_dim, 1, 3)

        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

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
        out_dense, out_sparse = outputs['dense'], outputs['sparse']

        split_map = outputs['split_map_raw'].cpu()
        split_map_dense = split_map[:, 0]
        split_map_middle = split_map[:, 1]
        split_map_sparse = split_map[:, 2]

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


        def get_pred_count(pred_logits):
            if len(pred_logits) ==0:
                return 0
            outputs_scores = torch.nn.functional.softmax(pred_logits.unsqueeze(0), -1)[..., 1][0]
            return outputs_scores.shape[0]

        # 计算mae和mse
        count_dense_list = []
        count_middle_list = []
        count_sparse_list = []
        if out_sparse is not None:
            sparse_mask = F.interpolate(split_map_sparse.unsqueeze(1).float(), size=out_sparse['fea_shape']).squeeze(1).bool().flatten(-2)
            mask_sparse = index_sparse & sparse_mask
            pred_logits_sparse = out_sparse['pred_logits'][mask_sparse]
            count_sparse0 = get_pred_count(pred_logits_sparse)
            count_sparse_list.append(count_sparse0)

            middle_mask = F.interpolate(split_map_middle.unsqueeze(1).float(), size=out_sparse['fea_shape']).squeeze(1).bool().flatten(-2)
            mask_middle = index_sparse & middle_mask
            pred_logits_middle = out_sparse['pred_logits'][mask_middle]
            count_middle0 = get_pred_count(pred_logits_middle)
            count_middle_list.append(count_middle0)

        if out_dense is not None:
            dense_mask = F.interpolate(split_map_dense.unsqueeze(1).float(), size=out_dense['fea_shape']).squeeze(1).bool().flatten(-2)
            mask_dense = index_dense & dense_mask
            pred_logits_dense = out_dense['pred_logits'][mask_dense]
            count_dense0 = get_pred_count(pred_logits_dense)
            count_dense_list.append(count_dense0)

            middle_mask = F.interpolate(split_map_middle.unsqueeze(1).float(), size=out_dense['fea_shape']).squeeze(1).bool().flatten(-2)
            mask_middle = index_dense & middle_mask
            pred_logits_middle = out_dense['pred_logits'][mask_middle]
            count_middle1 = get_pred_count(pred_logits_middle)
            count_middle_list.append(count_middle1)

        div_out['predict_cnt'] = 0
        if len(count_dense_list) > 0:
            div_out['predict_cnt'] += sum(count_dense_list) / len(count_dense_list)
        if len(count_middle_list) > 0:
            div_out['predict_cnt'] += sum(count_middle_list) / len(count_middle_list)
        if len(count_sparse_list) > 0 :
            div_out['predict_cnt'] += sum(count_sparse_list) / len(count_sparse_list)

        gt_split_map = 1 - (torch.cat([tgt['seg_level_map'] for tgt in kwargs['targets']], dim=0) > self.seg_level_split_th).long()
        div_out['gt_split_map'] = gt_split_map
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
        if self.use_seg_head:
            seg_map = self.seg_head(fea_x8)  # 已经经过sigmoid处理了
            pred_seg_map_4x = F.interpolate(seg_map, size=features['4x'].tensors.shape[-2:])
            pred_seg_map_8x = F.interpolate(seg_map, size=features['8x'].tensors.shape[-2:])
            features['4x'].tensors = features['4x'].tensors * pred_seg_map_4x
            features['8x'].tensors = features['8x'].tensors * pred_seg_map_8x
            outputs['seg_head_map'] = seg_map

        # apply quadtree splitter
        bs, _, src_h, src_w = fea_x8.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map_logits = self.quadtree_splitter(fea_x8)
        split_map = self.get_split_masks(split_map_logits, 3)
        split_map_dense = split_map[:, 0] | split_map[:, 1]
        split_map_dense = F.interpolate(split_map_dense.unsqueeze(1).float(), size=(ds_h, ds_w)).squeeze(1).bool()
        split_map_dense = split_map_dense.flatten(-2).to(split_map_logits.device)
        split_map_sparse = split_map[:, 1] | split_map[:, 2]
        split_map_sparse = F.interpolate(split_map_sparse.unsqueeze(1).float(), size=(sp_h, sp_w)).squeeze(1).bool()
        split_map_sparse = split_map_sparse.flatten(-2).to(split_map_logits.device)

        # context encoding
        def get_encode_context(encoder, feats_name, stride):

            src, mask = features[feats_name].decompose()
            src_pos_embed = pos[feats_name]
            assert mask is not None

            encode_src = encoder(src, src_pos_embed, mask, **kwargs)

            query_embed, points_queries, query_feats = self.points_queris_embed(samples, stride=stride, src=encode_src, **kwargs)
            context_info = (encode_src, src_pos_embed, mask, points_queries, query_embed)

            return context_info


        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:

            context_info_8x = get_encode_context(self.context_encoder_8x, "8x", 8)
            encode_src, src_pos_embed, mask, anchor_points, points_embed = context_info_8x

            # quadtree layer0 forward (sparse)
            hs = encode_src.flatten(2).permute(0, 2, 1)
            # if 'test' in kwargs:
            #     mask = (split_map_sparse > 0.5).cpu()
            #     hs = hs[mask].unsqueeze(0)
            #     anchor_points = anchor_points.unsqueeze(0)[mask]
            outputs_sparse = self.predict(self.class_embed_8x, self.coord_embed_8x, self.head_size_8x, samples, anchor_points, hs, 8, **kwargs)
            outputs_sparse['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            outputs_sparse['fea_shape'] = features['8x'].tensors.shape[-2:]
        else:
            outputs_sparse = None
        
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            # quadtree layer1 forward (dense)
            context_info_4x = get_encode_context(self.context_encoder_4x, "4x", 4)
            encode_src, src_pos_embed, mask, anchor_points, points_embed = context_info_4x

            hs = encode_src.flatten(2).permute(0, 2, 1)
            # if 'test' in kwargs:
            #     mask = (split_map_dense > 0.5).cpu()
            #     hs = hs[mask].unsqueeze(0)
            #     anchor_points = anchor_points.unsqueeze(0)[mask]
            outputs_dense = self.predict(self.class_embed_4x, self.coord_embed_4x, self.head_size_4x, samples, anchor_points, hs, 4, **kwargs)
            outputs_dense['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            outputs_dense['fea_shape'] = features['4x'].tensors.shape[-2:]
        else:
            outputs_dense = None
        
        # format outputs
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_logits'] = split_map_logits
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs

    def get_split_masks(self, out, n):
        # 获取每个像素的类别索引
        class_indices = torch.argmax(out, dim=1)  # 形状为 (1, 256, 256)

        # 初始化掩码张量
        b, h, w = class_indices.shape
        masks = torch.zeros(b, n, h, w, dtype=torch.uint8)  # 形状为 (n, 256, 256)

        # 为每个类别生成掩码
        for i in range(n):
            masks[:, i] = (class_indices == i)

        # 可选：转换为布尔掩码
        masks = masks.bool()
        return masks

    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']

        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride // 2 - 1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1, 0)  # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        return query_embed, points_queries, query_feats

    def predict(self, class_embed, coord_embed, head_size, samples, points_queries, hs, pq_stride, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (coord_embed(hs).sigmoid() - 0.5) * 2.0

        # headsizes
        ouputs_head_sizes = head_size(hs)

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[..., 0] /= (img_h / 256)
            outputs_offsets[..., 1] /= (img_w / 256)

        outputs_points = outputs_offsets + points_queries
        out = {
            'pred_logits': outputs_class,
            'pred_points': outputs_points,
            'img_shape': img_shape,
            'pred_offsets': outputs_offsets,
            'pred_head_sizes': ouputs_head_sizes
        }

        out['points_queries'] = points_queries
        out['pq_stride'] = pq_stride
        return out

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = self.warmup_ep

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse, _ = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense, _ = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse, _ = criterion(output_sparse, targets)
            loss_dict_dense, _ = criterion(output_dense, targets)

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

        # splitter depth loss
        pred_seg_levels = outputs['split_map_logits'].cpu()
        gt_seg_levels = []
        for tgt in targets:
            gt_seg_level = F.adaptive_avg_pool2d(tgt['seg_level_map'], pred_seg_levels.shape[-2:])
            seg_level = torch.ones(gt_seg_level.shape, device=gt_seg_level.device)
            seg_level[gt_seg_level < (self.seg_level_split_th - self.seg_level_split_range)] = 2
            seg_level[gt_seg_level > (self.seg_level_split_th + self.seg_level_split_range)] = 0
            gt_seg_levels.append(seg_level)
        gt_seg_levels = torch.cat(gt_seg_levels, dim=0).long().cpu()
        loss_split_seg_level = self.ce_loss(pred_seg_levels.float(), gt_seg_levels)
        loss_split = loss_split_seg_level
        losses += loss_split * 0.1
        loss_dict['loss_seg_level_map'] = loss_split * 0.1

        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}


def build_pet(args, backbone, num_classes):
    # build model
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )
    return model
