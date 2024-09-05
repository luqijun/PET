import torch

from .backbones import *
from .transformer import *
from .pet import build_pet

from .loss import build_criterion

def build_model(args):
    device = torch.device(args.device)

    # build backbone
    num_classes = 1
    args.num_classes = num_classes
    backbone = build_backbone_vgg(args)

    # build model
    model_name = args.get('model', 'pet')
    match model_name:
        case "pet": model = build_pet(args, backbone, num_classes)
        case _:
            raise ValueError(f"Cannot find model {model_name}")

    # build loss criterion
    criterion = build_criterion(args)
    criterion.to(device)

    return model, criterion

