import torch

from .backbones import *
from .loss import build_criterion
from .pet import build_pet
from .pet_dialated import build_pet as build_pet_dialated
from .pet_dialated_full_split_dec import build_pet as build_pet_dialated_full_split_dec
from .pet_dialated_full_split_dec_ablation_dense import build_pet as build_pet_dialated_full_split_dec_ablation_dense
from .pet_dialated_full_split_dec_ablation_sparse import build_pet as build_pet_dialated_full_split_dec_ablation_sparse
from .pet_dialated_full_split_dec_ablation_sparse_qnum import \
    build_pet as build_pet_dialated_full_split_dec_ablation_sparse_qnum
from .pet_dialated_full_split_dec_sparse import build_pet as build_pet_dialated_full_split_dec
from .pet_dialated_full_split_dec_sparse import build_pet as build_pet_dialated_full_split_dec_sparse
from .pet_dialated_swin import build_pet as build_pet_dialated_swin
from .pet_dialated_swin_enc import build_pet as build_pet_dialated_swin_enc
from .pet_dialated_swin_enc_dec import build_pet as build_pet_dialated_swin_enc_dec
from .pet_encoder_only import build_pet as build_pet_encoder_only
from .pet_encoder_only_dense_merge import build_pet as build_pet_encoder_only_dense_merge
from .pet_encoder_only_middle_merge import build_pet as build_pet_encoder_only_middle_merge
from .pet_eomm_head_size import build_pet as build_pet_eomm_head_size
from .pet_eomm_multi_points import build_pet as build_pet_eomm_multi_points
from .pet_error_map import build_pet as build_pet_error_map
from .transformer import *


def build_model(args):
    device = torch.device(args.device)

    # build backbone
    num_classes = 1
    args.num_classes = num_classes
    backbone = build_backbone_vgg(args)

    # build model
    model_name = args.get('model', 'pet')
    match model_name:
        case "pet":
            model = build_pet(args, backbone, num_classes)
        case "pet_dialated":
            model = build_pet_dialated(args, backbone, num_classes)
        case "pet_dialated_full_split_dec":
            model = build_pet_dialated_full_split_dec(args, backbone, num_classes)
        case "pet_dialated_full_split_dec_sparse":
            model = build_pet_dialated_full_split_dec_sparse(args, backbone, num_classes)
        case "pet_dialated_full_split_dec_ablation_sparse":
            model = build_pet_dialated_full_split_dec_ablation_sparse(args, backbone, num_classes)
        case "pet_dialated_full_split_dec_ablation_dense":
            model = build_pet_dialated_full_split_dec_ablation_dense(args, backbone, num_classes)
        case "pet_dialated_full_split_dec_ablation_sparse_qnum":
            model = build_pet_dialated_full_split_dec_ablation_sparse_qnum(args, backbone, num_classes)
        case "pet_dialated_swin":
            model = build_pet_dialated_swin(args, backbone, num_classes)
        case "pet_dialated_swin_enc":
            model = build_pet_dialated_swin_enc(args, backbone, num_classes)
        case "pet_dialated_swin_enc_dec":
            model = build_pet_dialated_swin_enc_dec(args, backbone, num_classes)
        case "pet_error_map":
            model = build_pet_error_map(args, backbone, num_classes)
        case "pet_encoder_only":
            model = build_pet_encoder_only(args, backbone, num_classes)
        case "pet_encoder_only_dense_merge":
            model = build_pet_encoder_only_dense_merge(args, backbone, num_classes)
        case "pet_encoder_only_middle_merge":
            model = build_pet_encoder_only_middle_merge(args, backbone, num_classes)
        case "pet_eomm_head_size":
            model = build_pet_eomm_head_size(args, backbone, num_classes)
        case "pet_eomm_multi_points":
            model = build_pet_eomm_multi_points(args, backbone, num_classes)
        case _:
            raise ValueError(f"Cannot find model {model_name}")

    # build loss criterion
    criterion = build_criterion(args)
    criterion.to(device)

    return model, criterion
