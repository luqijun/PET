import os

import matplotlib.pyplot as plt
import numpy as np
# from transformers import pipeline
from PIL import Image
from scipy.io import loadmat

from depth_anything_v2_metric import generate_depth_map, max_depth

use_metric_depth = True
if use_metric_depth:
    from depth_anything_v2_metric import (load_depth_model)

    depth_thresholds = [0.15, 0.2, 0.25, 0.3]
    depth_scale_factor = 3.0
else:
    from depth_anything_v2 import (load_depth_model)

    depth_thresholds = [100, 120, 140, 160]
    depth_scale_factor = 0.1

jump = False
min_value = 1e9
max_value = -1e9
use_double_knn_distance = False


def save_thresholded_depth_map_show(image, depth_map, save_path: str, title: str, thresholds):
    """
    根据给定的阈值绘制分割图，最后将原始图像和分割图放在一行上显示。
    """

    # 创建一个1行(len(thresholds) + 2)列的子图布局
    fig, axs = plt.subplots(1, len(thresholds) + 2, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f0f0')  # 设置整个图形的背景颜色

    # 原始图像
    axs[0].imshow(image)
    axs[0].axis('off')

    # 显示深度图像
    im = axs[1].imshow(depth_map, cmap='viridis', vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Depth', labelpad=5)
    # axs[1].set_title(title)
    axs[1].axis('off')

    # 对每个阈值进行分割并显示
    for i, threshold in enumerate(thresholds):
        # 使用阈值进行分割
        binary_image = depth_map > threshold

        # 显示分割后的图像
        axs[i + 2].imshow(binary_image, cmap='gray')
        axs[i + 2].set_title(f'Threshold: {threshold}')
        axs[i + 2].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def preprocess_show(path):
    depth_model = load_depth_model()
    min_head_size = 4
    knn_num = 3
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    filenames = ['IMG_5.jpg', 'IMG_29.jpg']

    for folder in ['train_data']:

        pet_folder = os.path.join(path, folder, 'pet_metric_depth' if use_metric_depth else 'pet')
        os.makedirs(pet_folder, exist_ok=True)

        images_folder = os.path.join(path, folder, 'images')

        # 标注文件夹
        annotations_folder = os.path.join(path, folder, 'ground-truth')

        for filename in filenames:

            image_path = os.path.join(images_folder, filename)
            print(image_path)
            image_pil = Image.open(image_path)
            points = \
                loadmat(os.path.join(annotations_folder, 'GT_' + filename.replace('.jpg', '.mat')))['image_info'][0][0][
                    0][
                    0][0]
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            if points.shape[1] <= 1:
                points = np.zeros((0, 2))
            points = points[:, 0:2]  # (x, y, ...)

            width, height = image_pil.size
            image = np.array(image_pil)

            plot_title = f"SHA Image {filename.split('.')[0]}"

            # 深度图
            image_depth_map = generate_depth_map(depth_model, image)
            if use_metric_depth:
                image_depth_map /= max_depth  # 归一化0到1
            depth_map_show_save_path = f'depth_show_{filename}'
            save_thresholded_depth_map_show(image, image_depth_map, depth_map_show_save_path, title=plot_title,
                                            thresholds=depth_thresholds)


if __name__ == '__main__':
    # root_test = '/mnt/c/Users/lqjun/Desktop'

    root = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final'
    preprocess_show(root)
    print('Process Success!')
    pass
