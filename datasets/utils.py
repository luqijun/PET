import torch
from util.misc import save_tensor_to_image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import torchvision.transforms as standard_transforms

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

restore_transform = standard_transforms.Compose([
    DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    standard_transforms.ToPILImage()
])

def visualization_loading_data(img, points, head_sizes, img_seg_level_map, img_seg_head_map, level_map_vmin=0.0, level_map_max=1.0):

    # 将 image tensor 转换为 PIL Image 对象
    image_pil = restore_transform(img)
    draw = ImageDraw.Draw(image_pil)
    for point, head_size in zip(points, head_sizes):
        y_lt, x_lt = point - head_size / 2
        y_rb, x_rb = point + head_size /2
        draw.rectangle([(x_lt, y_lt), (x_rb, y_rb)], outline='red') # 绘制红色矩形
        # draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red', outline='red')  # 绘制红色圆点

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f0f0')  # 设置整个图形的背景颜色
    axs[0].imshow(image_pil)
    axs[0].axis('off')
    axs[1].imshow(img_seg_level_map.squeeze(0).numpy(), cmap='viridis', vmin=level_map_vmin, vmax=level_map_max) #
    axs[1].axis('off')
    axs[2].imshow(img_seg_head_map.squeeze(0).numpy(), cmap='gray')
    axs[2].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def compute_density(points):
    """
    Compute crowd density:
        - defined as the average nearest distance between ground-truth points
    """
    points_tensor = points
    dist = torch.cdist(points_tensor, points_tensor, p=2)
    if points_tensor.shape[0] > 1:
        dist_sort = dist.sort(dim=1)[0]
        density = dist_sort[:, 1].mean().reshape(-1)
        nearest_num = min(points_tensor.shape[0], 4)
        knn_distances = dist_sort[:, 1:nearest_num].mean(dim=1)
    else:
        density = torch.tensor(999.0).reshape(-1)
        knn_distances = torch.zeros((points_tensor.shape[0],))
    return density, knn_distances


def random_crop(img, points, head_sizes, seg_level_map, seg_head_map, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_seg_level_map = seg_level_map[:, start_h:end_h, start_w:end_w]
    result_seg_head_map = seg_head_map[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    result_head_sizes = head_sizes[idx]

    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_seg_level_map = torch.nn.functional.interpolate(result_seg_level_map.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_seg_head_map = torch.nn.functional.interpolate(result_seg_head_map.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    result_head_sizes *= fH * fW
    return result_img, result_points, result_head_sizes, result_seg_level_map, result_seg_head_map


min_depth_weight = 1e9
max_depth_weight = -1e9
def cal_match_weight_by_depth(depth_values, scale, head_size_weight = 0.5):

    if depth_values.shape[1] == 0:
        return depth_values

    depth_values = depth_values * 0.1 * scale # 人头大小的一半
    depth_values = head_size_weight * torch.clamp(depth_values, min=4.0) # 最小人头为4
    weights = 1 / depth_values

    weights = torch.clamp(weights, min=0.01, max=0.09)

    global min_depth_weight, max_depth_weight
    min_w, max_w = torch.min(weights, dim=1)[0], torch.max(weights, dim=1)[0] # weights.min(dim=1), weights.max(dim=1)
    if min_w < min_depth_weight:
        min_depth_weight = min_w
    if max_w > max_depth_weight:
        max_depth_weight = max_w

    return weights


def cal_match_weight_by_headsizes(head_sizes, head_size_weight = 1.0, min=0.01, max=0.09):

    if len(head_sizes) == 0:
        return head_sizes
    weights = 1 / (head_sizes * head_size_weight)
    weights = torch.clamp(weights, min=min, max=max)
    return weights.unsqueeze(0)


def sample_points_in_range(points, sizes, m=3):
    """
    points: Tensor of shape (n, 2) representing n points in 2D space.
    sizes: Tensor of shape (n,) representing the influence range for each point.
    m: Number of points to sample for each point.

    Returns:
    new_points: Tensor of shape (n * m, 2) representing the new sampled points.
    """
    if len(points) == 0 or m==0:
        return points

    n = points.shape[0]

    # Generate random offsets for each point
    # Shape: (n, m, 2)
    random_offsets = torch.rand((n, m, 2), device=points.device) * 2 - 1  # Random values in range [-1, 1]
    random_offsets = random_offsets * sizes.view(n, 1, 1)  # Scale by the influence range

    # Apply offsets to the original points
    # Shape: (n, m, 2)
    points = points.unsqueeze(1)
    new_points = points + random_offsets

    # 拼接原有的点
    new_points = torch.cat([points, new_points], dim=1)

    # Flatten the new points to shape (n * m, 2)
    new_points = new_points.view(-1, 2)

    return new_points


def sample_points_around(points, sizes):
    # 确保输入的形状正确
    assert points.shape[1] == 2, "points must have shape (n, 2)"
    assert sizes.shape[0] == points.shape[0], "sizes must have shape (n,)"

    n = points.shape[0]

    # 生成四个方向的随机偏移量
    offsets = torch.rand(n, 4, 2) * sizes.view(-1, 1, 1)

    # 定义四个方向的偏移量方向
    directions = torch.tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=points.dtype, device=points.device)

    # 计算每个方向的偏移量
    points = points.view(n, 1, 2)
    sampled_points = points + offsets * directions

    sampled_points = torch.cat([points, sampled_points], dim=1)

    # 将结果展平为 (n*5, 2) 的形状
    sampled_points = sampled_points.view(-1, 2)

    return sampled_points


if __name__ == "__main__":
    # Example usage:
    # points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # sizes = torch.tensor([0.5, 1.0, 0.3])
    # m = 3
    #
    # new_points = sample_points_in_range(points, sizes, m)
    # print(new_points)

    # 示例使用
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    sizes = torch.tensor([0.5, 0.3])

    sampled_points = sample_points_around(points, sizes)
    print(sampled_points)