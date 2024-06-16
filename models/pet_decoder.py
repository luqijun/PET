"""
PET Decoder model
"""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import *
from .transformer import *
from util.nms import get_boxes_from_depths

class PETDecoder(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 1, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        pqs = self.get_point_query(samples, features, **kwargs)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        outputs['query_shape'] = pqs[2].shape[-2:]
        return outputs

    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats, depth_embed = self.points_queris_embed(samples, self.pq_stride,
                                                                                             src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2, 0, 1)  # NxCxHxW --> (HW)xNxC
            depth_embed = depth_embed.flatten(2).permute(2, 0, 1)
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx, depth_embed = self.points_queris_embed_inference(samples,
                                                                                                              self.pq_stride,
                                                                                                              src,
                                                                                                              **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx, depth_embed)
        return out

    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        depth_input_embed = kwargs['depth_input_embed'] # B, C, H, W
        dense_input_embed = kwargs['dense_input_embed']
        
        if 'level_embed' in kwargs:
            level_embed = kwargs['level_embed'].view(1, -1, 1, 1)
            dense_input_embed = dense_input_embed + level_embed
        
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down,shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        # depth_embed
        depth_embed = depth_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = depth_embed.shape[:2]
        depth_embed = depth_embed.view(bs, c, h, w)

        return query_embed, points_queries, query_feats, depth_embed
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # dense position encoding at every pixel location
        depth_input_embed = kwargs['depth_input_embed']  # B, C, H, W
        dense_input_embed = kwargs['dense_input_embed']
        
        if 'level_embed' in kwargs:
            level_embed = kwargs['level_embed'].view(1, -1, 1, 1)
            dense_input_embed = dense_input_embed + level_embed
        
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + stride//2 -1) // stride

        # generate points queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get points queries embedding 
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        depth_embed = depth_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        # window-rize
        query_embed = query_embed.reshape(bs, c, h, w)
        depth_embed = depth_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        depth_embed_win = window_partition(depth_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # dynamic point query generation
        div = kwargs['div']
        thrs = 0.5
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > thrs).sum(dim=0)[:,0]
        v_idx = valid_div > 0
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        depth_embed_win = depth_embed_win[:, v_idx]
        points_queries_win = points_queries_win.to(v_idx.device)
        points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)
    
        return query_embed_win, points_queries_win, query_feats_win, v_idx, depth_embed_win

    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """

        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape

        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0
        # outputs_depths = self.depth_embed(hs).sigmoid()[-1]

        # generate bounding boxes
        bs = samples.tensors.shape[0]
        center_points = points_queries.to(outputs_class.device)
        center_points = torch.repeat_interleave(center_points.unsqueeze(0), repeats=bs, dim=0) # (bs, num_queries, 2)

        base_size = 60
        scales = torch.tensor([tgt['scale'] for tgt in kwargs['targets']], device=outputs_class.device) # (bs,)
        depth_values = []
        depth_maps = torch.cat([tgt['depth'] for tgt in kwargs['targets']], dim=0)
        for c_points, d_map in zip(center_points, depth_maps):
            d_value = d_map[c_points[:, 0], c_points[:, 1]]
            depth_values.append(d_value)
        depth_values = torch.stack(depth_values, dim=0).to(center_points.device) # (bs, num_priors)

        # scales0 = scales
        # scales1 = scales * 1.5
        # scales2 = scales * 0.5
        # anchor_bboxes0 = get_boxes_from_depths(center_points, depth_values, scale=scales0.unsqueeze(-1), img_h=img_h, img_w=img_w) # (bs, num_priors, 4)
        # anchor_bboxes1 = get_boxes_from_depths(center_points, depth_values, scale=scales1.unsqueeze(-1), img_h=img_h, img_w=img_w) # (bs, num_priors, 4)
        # anchor_bboxes2 = get_boxes_from_depths(center_points, depth_values, scale=scales2.unsqueeze(-1), img_h=img_h, img_w=img_w, min_size=12.0 * 0.5) # (bs, num_priors, 4)
        # anchor_bboxes = torch.cat([anchor_bboxes0, anchor_bboxes1, anchor_bboxes2], dim=1)

        anchor_bboxes = get_boxes_from_depths(center_points, depth_values, scale=scales.unsqueeze(-1), img_h=img_h, img_w=img_w)


        # normalize point-query coordinates
        points_queries = points_queries.float().to(outputs_class.device)
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {
            'pred_logits': outputs_class[-1],
            'pred_points': outputs_points,
            'img_shape': img_shape,
            'pred_offsets': outputs_offsets[-1],
            # 'pred_depth':outputs_depths,
            'anchor_bboxes':anchor_bboxes,
            'anchor_points': center_points,
        }
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

