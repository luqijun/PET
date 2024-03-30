import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as io


def load_point_values(img_depth_path, gt_points_path):

    # load the images
    img_depth = Image.open(img_depth_path)
    img_depth = np.array(img_depth)
    points = io.loadmat(gt_points_path)['image_info'][0][0][0][0][0][:,::-1]
    values = img_depth[points[:, 0].astype(int), points[:, 1].astype(int)]
    return values / 255

def get_img_pixels(root_dir):

    flattened_pixels = []
    for folder in ['train_data', 'test_data']:
        depth_folder_path = os.path.join(root_dir, folder, 'images_depth')
        gt_folder_path = os.path.join(root_dir, folder, 'ground-truth')
        for img_path in os.listdir(depth_folder_path):
            img_depth_path = os.path.join(depth_folder_path, img_path)
            gt_points_path = os.path.join(gt_folder_path, 'GT_' + img_path.replace('.jpg', '.mat'))
            values = load_point_values(img_depth_path, gt_points_path)
            flattened_pixels.append(values)

    all_pixels = np.concatenate(flattened_pixels)
    return all_pixels


# 读取图像列表
root_dir = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final'
all_pixels = get_img_pixels(root_dir)

# 统计像素值分布
# 使用直方图显示分布
plt.hist(all_pixels, bins=5, range=[0, 1])
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution across all Images')
plt.show()