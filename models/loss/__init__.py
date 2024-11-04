import torch

from .criterion import SetCriterion
from .criterion_4x1 import SetCriterion as SetCriterion_4x1
from .criterion_ignore_pt_loss import SetCriterion as SetCriterion_ignore_pt_loss
from .criterion_ignore_pt_loss_split import SetCriterion as SetCriterion_ignore_pt_loss_split
from .criterion_middle_merge import SetCriterion as SetCriterion_Middle_Merge
from .criterion_mm_head_size import SetCriterion as SetCriterion_mm_head_size
from .criterion_multi_points import SetCriterion as SetCriterion_multi_points
from .matcher import build_matcher
from .matcher_4x1 import build_matcher as build_matcher_4x1
from .matcher_head_size import build_matcher as build_matcher_head_size
from .matcher_preselect import build_matcher as build_matcher_pre_select
from .matcher_preselect_with_knndist import build_matcher as build_matcher_pre_select_with_knndist
from .matcher_with_points_weight import build_matcher as build_matcher_with_points_weight


def build_criterion(args):
    device = torch.device(args.device)
    # build loss criterion
    matcher_name = args.get("matcher", "matcher_depth")
    match matcher_name:
        case "matcher":
            matcher = build_matcher(args)
        case "matcher_4x1":
            matcher = build_matcher_4x1(args)
        case "matcher_with_points_weight":
            matcher = build_matcher_with_points_weight(args)
        case "matcher_pre_select":
            matcher = build_matcher_pre_select(args)
        case "matcher_preselect_with_knndist":
            matcher = build_matcher_pre_select_with_knndist(args)
        case "matcher_head_size":
            matcher = build_matcher_head_size(args)
        case _:
            raise ValueError(f"Matcher Name not find：{matcher_name}")

    if "weight_dict" in args:
        weight_dict = args.weight_dict
        losses = args.losses
    else:
        weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
        losses = ['labels', 'points']

    criterion_name = args.get("criterion", "SetCriterion")
    match criterion_name:
        case "SetCriterion":
            criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                     eos_coef=args.eos_coef, losses=losses)
        case "criterion_4x1":
            criterion = SetCriterion_4x1(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                         eos_coef=args.eos_coef, losses=losses)
        case "criterion_middle_merge":
            criterion = SetCriterion_Middle_Merge(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                                  eos_coef=args.eos_coef, losses=losses)
        case "criterion_mm_head_size":
            criterion = SetCriterion_mm_head_size(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                                  eos_coef=args.eos_coef, losses=losses)
        case "criterion_multi_points":
            criterion = SetCriterion_multi_points(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                                  eos_coef=args.eos_coef, losses=losses, args=args)
        case "criterion_ignore_pt_loss":
            criterion = SetCriterion_ignore_pt_loss(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                                    eos_coef=args.eos_coef, losses=losses, args=args)
        case "criterion_ignore_pt_loss_split":
            criterion = SetCriterion_ignore_pt_loss_split(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                                          eos_coef=args.eos_coef, losses=losses, args=args)
        case _:
            raise ValueError(f"Criterion Name not find：{criterion_name}")
    criterion.to(device)
    return criterion
