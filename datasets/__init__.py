# from .SHB import build as build_shb
# from .UCF_QNRF import build as build_ucf_qnrf
from .JHU import build as build_jhu
from .JHU_General import build as build_jhu_general
from .SHA import build as build_sha
from .SHA_FgMask import build as build_sha_fg_mask
from .SHA_General import build as build_sha_general
from .SHA_General_Multi_Points import build as build_sha_general_multi_points
from .SHB_General import build as build_shb_general


# data_path = {
#     'SHA': '',
#     'SHB': './data/ShanghaiTech/part_B/',
#     'UCF_QNRF': './data/ShanghaiTech/UCF_QNRF/',
#     'JHU': './data/JHU/',
# }

def build_dataset(image_set, args):
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == "SHA_FgMask":
        return build_sha_fg_mask(image_set, args)
    if args.dataset_file == "SHA_General":
        return build_sha_general(image_set, args)
    if args.dataset_file == "SHA_General_Multi_Points":
        return build_sha_general_multi_points(image_set, args)
    if args.dataset_file == 'SHB_General':
        return build_shb_general(image_set, args)
    # if args.dataset_file == 'UCF_QNRF':
    #     return build_ucf_qnrf(image_set, args)
    if args.dataset_file == 'JHU':
        return build_jhu(image_set, args)
    if args.dataset_file == 'JHU_General':
        return build_jhu_general(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
