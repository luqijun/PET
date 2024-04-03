import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .SHB import build as build_shb
from .UCF_QNRF import build as build_ucf_qnrf
from .JHU import build as build_jhu

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'SHB': './data/ShanghaiTech/part_B/',
    'UCF_QNRF': './data/ShanghaiTech/UCF_QNRF/',
    'JHU': './data/ShanghaiTech/JHU/',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'SHB':
        return build_shb(image_set, args)
    if args.dataset_file == 'UCF_QNRF':
        return build_ucf_qnrf(image_set, args)
    if args.dataset_file == 'JHU':
        return build_jhu(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
