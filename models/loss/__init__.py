from .criterion import *
from .matcher import build_matcher
from .matcher_pre_select import build_matcher as build_matcher_pre_select

def build_criterion(args):
    device = torch.device(args.device)
    # build loss criterion
    matcher_name = args.get("matcher", "matcher")
    match matcher_name:
        case "matcher": matcher = build_matcher(args)
        case "matcher_pre_select": matcher = build_matcher_pre_select(args)
        case _:
            raise ValueError("Matcher Name not find")

    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return criterion