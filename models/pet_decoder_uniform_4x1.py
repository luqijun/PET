"""
PET Decoder model
"""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import *
from .transformer import *

class PETDecoder(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.num_predict_heads = self.args.num_predict_heads
        self.return_pred_layers_idx = self.args.return_pred_layers_idx
        self.class_embed_list = nn.Sequential(*[nn.Linear(hidden_dim, num_classes + 1) for _ in range(self.num_predict_heads)])
        self.coord_embed_list = nn.Sequential(*[MLP(hidden_dim, hidden_dim, 2, 3) for _ in range(self.num_predict_heads)])

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        # (query_embed, points_queries, query_feats, depth_embed)
        pqs = self.get_point_query(samples, features, **kwargs)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs_list, v_idx_list = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # # prediction
        points_queries = pqs[1]
        # if len(v_idx_list) > 0:
        #     qH, qW = pqs[2].shape[-2:]
        #     dec_win_h, dec_win_w = kwargs['dec_win_size_list'][-1]
        #     points_queries = points_queries.reshape(qH, qW, 2).permute(2, 0, 1).unsqueeze(0)
        #     points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        #     points_queries_win = points_queries_win.to(encode_src.device)
        #     points_queries = points_queries_win[:, v_idx_list[-1]].reshape(-1, 2)

        outputs_list = []
        outputs_cache = {}
        for idx, (layer_idx) in enumerate(self.return_pred_layers_idx):
            if self.args.use_same_layer_idx and layer_idx in outputs_cache:
                outputs = outputs_cache[layer_idx]
            else:
                outputs = self.predict(samples, points_queries, hs_list[layer_idx], self.class_embed_list[idx], self.coord_embed_list[idx], **kwargs)
                outputs_cache[layer_idx] = outputs
            outputs['fea_shape']= features[self.feat_name].tensors.shape[-2:]
            outputs_list.append(outputs)
        return outputs_list

    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        query_embed, points_queries, query_feats, depth_embed = \
            self.points_queris_embed(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, depth_embed)
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

    def predict(self, samples, points_queries, hs, class_embed, coord_embed, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

