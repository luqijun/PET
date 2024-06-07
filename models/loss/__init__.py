from .criterion import *
from .matcher import build_matcher

def build_criterion(args):
    device = torch.device(args.device)
    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef, 'loss_ce_seg_head': 0.1 }
    losses = ['labels', 'points', 'segheads']
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return criterion