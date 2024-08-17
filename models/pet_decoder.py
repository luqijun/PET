"""
PET Decoder model
"""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import *
from .transformer import *
from timm.models.layers import trunc_normal_
import math

class PETDecoder(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim
        self.embed_dims = hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.output_count = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'

        self.as_two_stage = True
        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims,
                                          self.embed_dims)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims)

            self.enc_class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.enc_coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.conv_en = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1)

    def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        if not self.as_two_stage:
            pqs = self.get_point_query(samples, features, **kwargs)
            # point querying
            kwargs['pq_stride'] = self.pq_stride
            hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)
            # prediction
            points_queries = pqs[0]
            outputs = self.predict(samples, points_queries, hs, **kwargs)
            outputs['dec_win_size'] = kwargs['dec_win_size']
            outputs['fea_shape'] = encode_src.shape[-2:]
            return outputs
        else:
            encode_src_up = F.interpolate(encode_src, scale_factor=2.0)
            encode_src_up = torch.cat([encode_src_up, features['4x'].tensors], dim=1)
            encode_src_up = self.conv_en(encode_src_up)

            output_proposals, output_memory = self.gen_encoder_output_proposals(samples, encode_src_up, **kwargs)
            enc_outputs_class = self.enc_class_embed(output_memory)
            enc_outputs_coord_unact = self.enc_coord_embed(output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            enc_outputs_coord = torch.flip(enc_outputs_coord, dims=[-1])

            enc_outputs_class_scores = torch.softmax(enc_outputs_class, dim=-1)[..., 1]
            num_queries = int(output_memory.shape[1] * 0.9)
            topk_proposals = torch.topk(enc_outputs_class_scores, num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 2))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid() # (B, N, 2) (w, h)
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            # query_pos, query = torch.split(pos_trans_out, self.embed_dims, dim=2)
            query_pos_tran = pos_trans_out
            # dense_input_embed = kwargs['dense_input_embed']

            # 构建rectangle
            B, C, H, W = encode_src_up.shape
            # query_mask = torch.full((B, H, W), False, device=encode_src.device).flatten(1)
            # query_mask.scatter_(dim=1, index=topk_proposals, src=torch.full_like(topk_proposals, True, dtype=torch.bool, device=encode_src.device) )
            # query_mask = query_mask.reshape(B, H, W)
            sampling_grids = 2 * reference_points - 1
            query_sample = F.grid_sample(encode_src_up, sampling_grids.unsqueeze(-2)).squeeze(-1).permute(0, 2, 1)

            reference_points_pos = torch.stack([reference_points[..., 0] * W, reference_points[..., 1] * H], dim=-1)
            reference_points_pos = reference_points_pos.round().long()
            reference_points_pos = reference_points_pos[..., 1] * W + reference_points_pos[..., 0]
            reference_points_pos = torch.clamp(reference_points_pos, min=0, max=H*W-1)
            # reference_points_pos = reference_points_pos[..., 1]

            # query_sample = query_sample.detach()
            # query = encode_src_up.detach().flatten(2).permute(0, 2, 1) # 效果不太好
            query = torch.full((B, C, H, W), 0.0, device=encode_src.device).flatten(2).permute(0, 2, 1) # (B, H*W, C)
            query.scatter_(dim=1,
                           index=reference_points_pos.unsqueeze(-1).repeat(1, 1, C),
                           src=query_sample)

            # query_pos_sample = F.grid_sample(dense_input_embed, sampling_grids.unsqueeze(-2)).squeeze(-1).permute(0, 2, 1)
            query_pos = torch.full((B, C, H, W), 0.0, device=encode_src.device).flatten(2).permute(0, 2, 1)
            query_pos.scatter_(dim=1,
                           index=reference_points_pos.unsqueeze(-1).repeat(1, 1, C),
                           src=query_pos_tran)
            query_pos = query_pos.permute(1, 0, 2)

            query = query.permute(0, 2, 1).reshape(B, C, H, W)
            pqs = (reference_points, query, query_pos, None)

            # point querying
            kwargs['pq_stride'] = self.pq_stride
            hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)
            hs = torch.gather(hs, 2, reference_points_pos.unsqueeze(0).unsqueeze(-1).repeat(2, 1, 1, C))
            # prediction
            points_queries = pqs[0]
            outputs = self.predict(samples, points_queries, hs, **kwargs)
            outputs['dec_win_size'] = kwargs['dec_win_size']
            outputs['fea_shape'] = encode_src.shape[-2:]

            outputs['enc_outputs'] = {}
            outputs['enc_outputs']['pred_logits'] = enc_outputs_class
            outputs['enc_outputs']['pred_points'] = enc_outputs_coord
            outputs['enc_outputs']['img_shape'] = samples.tensors.shape[-2:]
            return outputs

    @staticmethod
    def get_proposal_pos_embed(proposals,
                               num_pos_feats: int = 128,
                               temperature: int = 10000):
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals.sigmoid() * scale
        # N, L, 2, 256
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 128, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, samples, encode_src, **kwargs):

        memory = encode_src
        memory_mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=memory.shape[-2:]).bool().flatten(1)

        bs = memory.size(0)
        proposals = []
        spatial_shapes = [memory.shape[-2:]]
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            # wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            # proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposal = grid.view(bs, -1, 2)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.001) & (output_proposals < 0.999)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory.flatten(-2).permute(0, 2, 1)
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]

        # h, w = memory.shape[-2:]
        # output_memory = output_memory.permute(0, 2, 1).reshape(bs, -1, h, w)
        return output_proposals, output_memory

    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride,
                                                                            src, **kwargs)
        query_embed = query_embed.flatten(2).permute(2, 0, 1)  # NxCxHxW --> (HW)xNxC

        out = (points_queries, query_feats, query_embed, None)
        return out

    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed'] # B, C, H, W
        
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

        return query_embed, points_queries, query_feats

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
                outputs_offsets[...,0] /= (img_h / 256)
                outputs_offsets[...,1] /= (img_w / 256)

            outputs_points = outputs_offsets[-1] + points_queries
        else:
            outputs_class = self.class_embed(hs)
            points_queries = torch.log(points_queries / (1 - points_queries)) # inverse sigmoid
            outputs_offsets = self.coord_embed(hs)
            outputs_points = (outputs_offsets[-1] + points_queries).sigmoid()
            outputs_points = torch.flip(outputs_points, dims=[-1])

        out = {
            'pred_logits': outputs_class[-1],
            'pred_points': outputs_points,
            'img_shape': img_shape,
            'pred_offsets': outputs_offsets[-1]
        }
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

