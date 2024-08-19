"""
PET Decoder model
"""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import *
from .transformer import *
from .layers.deformable_detr.deformable_transformer import DeformableTransformerDecoderWrap
from util.misc import inverse_sigmoid

class PETDecoder(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.decoder: DeformableTransformerDecoderWrap = kwargs['decoder']
        hidden_dim = args.hidden_dim
        self.embed_dims = hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
        self.as_two_stage = self.decoder.two_stage

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)

            self.pos_trans_fc = nn.Linear(self.embed_dims,
                                          self.embed_dims)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims)

            self.enc_class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.enc_coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)


    def forward(self, samples, features, context_info, **kwargs):

        outputs = {}
        memory, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = context_info
        if self.decoder.two_stage:

            fea_idx = 0 if self.feat_name == '4x' else 1
            h, w = spatial_shapes[fea_idx]
            start_idx = level_start_index[fea_idx]
            select_memory = memory[:, start_idx: start_idx + h * w]
            select_mask_flatten = mask_flatten[:, start_idx: start_idx + h * w]
            select_spatial_shapes = spatial_shapes[fea_idx:fea_idx + 1]

            # select_memory = memory
            # select_mask_flatten = mask_flatten
            # select_spatial_shapes = spatial_shapes

            output_memory, output_proposals = self.gen_encoder_output_proposals(select_memory, select_mask_flatten, select_spatial_shapes)
            enc_outputs_class = self.enc_class_embed(output_memory)
            enc_outputs_coord_unact = self.enc_coord_embed(output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()

            enc_outputs_class_scores = torch.softmax(enc_outputs_class, dim=-1)[..., 1]
            select_ratio = 0.2 if self.feat_name == '4x' else 0.2
            num_queries = int(output_memory.shape[1] * select_ratio)
            topk_proposals = torch.topk(enc_outputs_class_scores, num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 2))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()  # (B, N, 2) (w, h)
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            # query_pos, query = torch.split(pos_trans_out, self.embed_dims, dim=2)
            query_pos_tran = pos_trans_out

            # 构建query
            backbone_features = features[self.feat_name].tensors
            sampling_grids = 2 * reference_points - 1
            query_feats = F.grid_sample(backbone_features, sampling_grids.unsqueeze(-2)).squeeze(-1).permute(0, 2, 1).detach()

            query_embed = query_pos_tran
            points_queries = sampling_grids
            # dense_input_embed = kwargs['dense_input_embed']
            # query_embed = F.grid_sample(dense_input_embed, sampling_grids.unsqueeze(-2)).squeeze(-1).permute(0, 2, 1)
            # points_queries = sampling_grids

            div = kwargs['div']
            B, H, W = div.shape
            reference_points_pos = torch.stack([reference_points[..., 0] * W, reference_points[..., 1] * H], dim=-1)
            reference_points_pos = reference_points_pos.round().long()
            reference_points_pos = reference_points_pos[..., 1] * W + reference_points_pos[..., 0]
            reference_points_pos = torch.clamp(reference_points_pos, min=0, max=H * W - 1)

            div_new = []
            for div_f, idx in zip(div.flatten(1), reference_points_pos):
                div_new.append((div_f[idx]))
            div_new = torch.stack(div_new, dim=0).to(div.device)

            if 'test' in kwargs:
                mask = div_new > 0.5
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, query_feats.shape[-1])
                query_feats = torch.masked_select(query_feats, mask_expanded).view(B, -1, query_feats.shape[-1])
                query_embed = torch.masked_select(query_embed, mask_expanded).view(B, -1, query_embed.shape[-1])
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 2)
                points_queries = torch.masked_select(points_queries, mask_expanded).view(B, -1, points_queries.shape[-1])


            # 仅仅使用高级特征
            # low_h, low_w = spatial_shapes[1]
            # memory = memory[:, -low_h * low_w:]
            # lvl_pos_embed_flatten = lvl_pos_embed_flatten[:, -low_h * low_w:]
            # mask_flatten = mask_flatten[:, -low_h * low_w:]
            # valid_ratios = valid_ratios[:, 1:, :]
            # level_start_index = level_start_index[0]

            outputs['div_new'] = div_new
            outputs['enc_outputs'] = {}
            outputs['enc_outputs']['pred_logits'] = enc_outputs_class
            outputs['enc_outputs']['pred_points'] = torch.flip(enc_outputs_coord, dims=[-1])  # flip  (w, h) -> (h, w)
            outputs['enc_outputs']['img_shape'] = samples.tensors.shape[-2:]
            outputs['enc_outputs']['pq_stride'] = self.pq_stride
        else:

            # get points queries for transformer
            query_embed, points_queries, query_feats, v_idx, depth_embed = self.get_point_query(samples, features,
                                                                                                **kwargs)

            B = samples.tensors.shape[0]
            img_h, img_w = samples.tensors.shape[-2:]
            points_queries = points_queries.float().to(query_feats.device)
            points_queries = torch.flip(points_queries, dims=[-1])
            points_queries[:, 0] /= img_h
            points_queries[:, 1] /= img_w
            points_queries = points_queries.unsqueeze(0).expand(B, -1, -1)

            if 'train' in kwargs:
                query_feats = query_feats.flatten(2).permute(0, 2, 1)
            else:
                query_feats = query_feats.permute(1, 0, 2)
            query_embed = query_embed.permute(1, 0, 2)



        hs, inter_references_out = \
            self.decoder(query_feats, query_embed, points_queries, memory, mask_flatten, spatial_shapes, level_start_index, valid_ratios)

        # point querying
        kwargs['pq_stride'] = self.pq_stride

        # prediction
        dec_outputs = self.predict(samples, points_queries, hs, **kwargs)
        outputs.update(**dec_outputs)
        outputs['pq_stride'] = self.pq_stride
        return outputs

    def get_proposal_pos_embed(self, proposals):

        proposals = torch.flip(proposals, dims=[-1])

        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):

        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposal = grid.view(N_, -1, 2)
            proposals.append(proposal)

            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

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
            points_queries = points_queries.squeeze(1)

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
        points_queries = points_queries.to(query_feats.device)

        div = kwargs['div']
        thrs = 0.5
        valid_div = (div > thrs).flatten()

        query_embed = query_embed.flatten(2)[..., valid_div].permute(2, 0, 1)
        depth_embed = depth_embed.flatten(2)[..., valid_div].permute(2, 0, 1)
        points_queries = points_queries.flatten(2)[..., valid_div].permute(2, 0, 1)
        query_feats = query_feats.flatten(2)[..., valid_div].permute(2, 0, 1)
    
        return query_embed, points_queries, query_feats, None, depth_embed

    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        if not self.as_two_stage:
            outputs_class = self.class_embed(hs)
            # normalize to 0~1
            outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

            # normalize point-query coordinates
            points_queries = points_queries.float().cuda()
            points_queries[:, 0] /= img_h
            points_queries[:, 1] /= img_w

            # rescale offset range during testing
            if 'test' in kwargs:
                outputs_offsets[..., 0] /= (img_h / 256)
                outputs_offsets[..., 1] /= (img_w / 256)

            outputs_points = outputs_offsets[-1] + points_queries
        else:
            outputs_class = self.class_embed(hs)
            points_queries = inverse_sigmoid(points_queries)  # inverse sigmoid
            outputs_offsets = self.coord_embed(hs)
            outputs_points = (outputs_offsets[-1] + points_queries).sigmoid()
            outputs_points = torch.flip(outputs_points, dims=[-1])

        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

