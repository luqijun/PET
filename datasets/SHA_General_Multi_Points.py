import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional
from torch.utils.data import Dataset
from PIL import Image
import cv2
import h5py
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings
from util.misc import save_tensor_to_image
from .utils import (compute_density, random_crop,
                    cal_match_weight_by_depth, cal_match_weight_by_headsizes, sample_points_around,
                    visualization_loading_data)

warnings.filterwarnings('ignore')


class SHA(Dataset):
    def __init__(self, args, data_root, transform=None, train=False, flip=False):
        self.args = args
        self.root_path = data_root

        prefix = "train" if train else "test"
        self.prefix = prefix

        # get image and ground-truth list
        self.gt_list = {}
        prefix_list = ["train_data"] if prefix == "train" else ["test_data"]
        for sub_prefix in prefix_list:
            img_list = os.listdir(f"{data_root}/{sub_prefix}/images")
            for img_name in img_list:
                txt_name = img_name.replace(".jpg", ".txt")
                h5_name = img_name.replace(".jpg", ".h5")
                mat_name = img_name.replace(".jpg", ".mat")
                img_path = f"{data_root}/{sub_prefix}/images/{img_name}"
                self.gt_list[img_path] = {}
                self.gt_list[img_path]["points"] = f"{data_root}/{sub_prefix}/ground-truth/GT_{mat_name}"
                self.gt_list[img_path]["head_sizes"] = f"{data_root}/{sub_prefix}/pet_metric_depth/{self.head_sizes_folder}/{txt_name}"
                self.gt_list[img_path]["seg_level"] = f"{data_root}/{sub_prefix}/pet_metric_depth/{self.seg_level_folder}/{h5_name}"
                self.gt_list[img_path]["seg_head"] = f"{data_root}/{sub_prefix}/pet_metric_depth/{self.seg_head_folder}/{img_name}"

        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.pil_to_tensor = standard_transforms.ToTensor()
        self.train = train
        self.flip = flip
        self.patch_size = 256

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_paths = self.gt_list[img_path]
        points_path = gt_paths['points']
        head_sizes_path = gt_paths['head_sizes']
        seg_level_path = gt_paths['seg_level']
        seg_head_path = gt_paths['seg_head']

        img, points, head_sizes, img_seg_level_map, img_seg_head_map = load_data(img_path, points_path, head_sizes_path, seg_level_path, seg_head_path, self.train)
        points = points.float()

        # image transform
        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)
        scale = 1.0
        # random scale
        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # interpolation
            if scale * min_size > self.patch_size:
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0)
                points *= scale
                img_seg_level_map = F.interpolate(img_seg_level_map.unsqueeze(0), scale_factor=scale, mode="bilinear").squeeze(0)
                img_seg_head_map = F.interpolate(img_seg_head_map.unsqueeze(0), scale_factor=scale).squeeze(0)
                head_sizes *= scale

            # random crop patch
            img, points, head_sizes, img_seg_level_map, img_seg_head_map = random_crop(img, points, head_sizes, img_seg_level_map, img_seg_head_map, patch_size=self.patch_size)

            # random flip
            if random.random() > 0.5 and self.flip:
                img = torch.flip(img, dims=[2])
                points[:, 1] = self.patch_size - points[:, 1]
                img_seg_level_map = torch.flip(img_seg_level_map, dims=[2])
                img_seg_head_map = torch.flip(img_seg_head_map, dims=[2])

        # 可视化数据
        # visualization_loading_data(img, points, head_sizes, img_seg_level_map, img_seg_head_map)

        # target
        target = {}
        points = torch.Tensor(points)

        points = sample_points_around(points, head_sizes * 0.4)
        head_sizes = head_sizes.unsqueeze(-1).repeat(1, self.sample_points + 1).flatten(0)

        target['points'] = points
        target['labels'] = torch.ones([points.shape[0]]).long()

        # depth info
        h, w = img_seg_level_map.shape[-2:]
        h_coords = torch.clamp(points[:, 0].long(), min=0, max=h - 1)
        w_coords = torch.clamp(points[:, 1].long(), min=0, max=w - 1)
        depth = img_seg_level_map[:, h_coords, w_coords]
        target['seg_level_map'] = img_seg_level_map
        target['seg_head_map'] = img_seg_head_map
        target['head_sizes'] = head_sizes.float()
        target['match_point_weight'] = cal_match_weight_by_headsizes(head_sizes, self.args.head_size_weight) # cal_match_weight_by_depth(depth, scale, self.args.head_size_weight)
        # target['depth_weight'] = self.cal_depth_weight(depth, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        target['label_map'] = self.get_label_map(points, img_seg_level_map.shape[-2:])
        # if self.train:
        #     target['seg_map'] = self.get_seg_map_by_label_map(target['label_map'], 32, 32)

        # target['depth_encoding'] = self.encode_depth(img_depth, 8)
        # depth_weight = torch.clamp(1 - depth, min=0.1, max=0.9)
        # target['depth_weight'] = depth_weight / 8

        if self.train:
            density, knn_distances = compute_density(points)
            target['density'] = density
            target['knn_distances'] = knn_distances
        else:  # test
            target['image_path'] = img_path

        return img, target

    @property
    def sample_points(self):
        return self.args.sample_points

    @property
    def head_sizes_folder(self):
        return self.args.head_sizes_folder

    @property
    def seg_level_folder(self):
        return self.args.seg_level_folder

    @property
    def seg_head_folder(self):
        return self.args.seg_head_folder

    def get_label_map(self, points, shape):
        H, W = shape
        points = (points.round() - 1).long()
        points[:, 0] = torch.clamp(points[:, 0], max=H - 1, min=0)
        points[:, 1] = torch.clamp(points[:, 1], max=W - 1, min=0)

        result = torch.zeros(shape)
        result[points[:, 0], points[:, 1]] = 1
        return result


    def get_seg_map_by_label_map(self, label_map, region_H=16, region_W=16):

        H, W = label_map.shape
        label_map = label_map.clone().reshape(H // region_H, region_H, W // region_W, region_W).permute(0, 2, 1, 3)
        seg_map = label_map.flatten(-2).sum(-1)
        seg_map = (seg_map > 0).float()
        return seg_map

    def cal_depth_weight(self, depth_values, values):

        depth_values = depth_values / 1000

        num_parts = len(values) + 1
        # 划分为n个部分
        partitions = torch.linspace(0, 1, num_parts)  # 在（0，1）之间均匀划分为n个部分

        depth_weights = 1 - depth_values
        x = depth_weights

        # 初始化一个新的张量，用于存储结果
        result = torch.zeros_like(x)

        # 遍历每个部分
        for i in range(len(partitions) - 1):
            lower_bound = partitions[i]
            upper_bound = partitions[i + 1]
            mask = (x >= lower_bound) & (x < upper_bound)  # 创建一个布尔掩码
            result[mask] = values[i]  # 根据掩码赋值
        return result

    def encode_depth(self, img_depth, levels):

        # 划分为多个层级
        partitions = torch.linspace(0, 1, levels + 1)  # 在（0，1）之间均匀划分为n个部分
        result = torch.zeros_like(img_depth, device=img_depth.device)
        for i in range(len(partitions) - 1):
            lower_bound = partitions[i]
            upper_bound = partitions[i + 1]
            value = (partitions[i] + partitions[i + 1]) / 2
            mask = (img_depth >= lower_bound) & (img_depth < upper_bound)  # 创建一个布尔掩码
            result[mask] = 0.1 * value  # 根据掩码赋值

        return result


def load_data(img_path, points_path, head_sizes_path, seg_level_path, seg_head_path, train):

    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # load points
    points = io.loadmat(points_path)['image_info'][0][0][0][0][0][:, ::-1]

    # load head sizes
    head_sizes = np.loadtxt(head_sizes_path)

    # load seg level map
    with h5py.File(seg_level_path) as gt_seg_level_file:
        seg_level_map = np.array(gt_seg_level_file[list(gt_seg_level_file.keys())[0]])

    # load seg head map
    seg_head_map = cv2.imread(seg_head_path, cv2.COLOR_BGR2GRAY)
    seg_head_map = seg_head_map // 255
    # seg_head_map = torchvision.transforms.functional.to_tensor(seg_head_map)

    return img, torch.from_numpy(points.copy()), torch.from_numpy(head_sizes), \
        torch.from_numpy(seg_level_map).unsqueeze(0), torch.from_numpy(seg_head_map).unsqueeze(0)


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(args, data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = SHA(args, data_root, train=False, transform=transform)
        return val_set
