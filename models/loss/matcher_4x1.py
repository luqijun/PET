"""
Modules to compute bipartite matching
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from .utils import split_and_compute_cdist2

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    """
    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_point: This is the relative weight of the L2 error of the point coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, img_shape, pred_logits, pred_points, tgt_points, tgt_labels,match_point_weights, **kwargs):
        """ 
        Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """

        # flatten to compute the cost matrices in a batch
        out_prob = torch.cat(pred_logits, dim=0).softmax(-1)  # [batch_size * num_queries, 2]
        out_points_sizes = [p.shape[0] for p in pred_points]
        out_points = torch.cat(pred_points, dim=0)  # [batch_size * num_queries, 2]

        # concat target labels and points
        tgt_ids = torch.cat(tgt_labels, dim=0)
        tgt_points_sizes = [p.shape[0] for p in tgt_points]
        tgt_points = torch.cat(tgt_points, dim=0).float()
        match_point_weights = torch.cat(match_point_weights, dim=1)

        # compute the classification cost, i.e., - prob[target class]
        cost_class = -out_prob[:, tgt_ids]

        # compute the L2 cost between points
        img_h, img_w = img_shape
        out_points_abs = out_points.clone()
        out_points_abs[:,0] *= img_h
        out_points_abs[:,1] *= img_w
        cost_point = split_and_compute_cdist2(out_points_abs, out_points_sizes, tgt_points, tgt_points_sizes, p=2)
        # cost_point = torch.cdist(out_points_abs, tgt_points, p=2)

        # final cost matrix
        C = cost_point * match_point_weights + self.cost_class * cost_class
        # C = cost_point * self.cost_point + self.cost_class * cost_class
        # C = C.view(bs, num_queries, -1).cpu()

        indices = []
        src_length = 0
        tgt_length = 0
        for i, c_row in enumerate(C.split(tgt_points_sizes, -1)):
            c_row_cols = c_row.split(out_points_sizes, dim=0)
            indice = linear_sum_assignment(c_row_cols[i].cpu())
            indices.append((indice[0] + src_length, indice[1] + tgt_length))
            src_length += c_row_cols[i].shape[0]
            tgt_length += c_row_cols[i].shape[1]

        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
