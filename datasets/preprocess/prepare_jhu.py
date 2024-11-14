import os

import h5py
import numpy as np
# from transformers import pipeline
from PIL import Image
from tqdm import tqdm

from prepare_base import handle_depth_var_split, handle_var_split, handle_depth_split, \
    handle_depth_map, handle_density_level_map, handle_density_map
from utils import (resize_image_and_points)

use_metric_depth = True
if use_metric_depth:
    from depth_anything_v2_metric import (load_depth_model)

    depth_thresholds = [0.15, 0.2, 0.25, 0.3]
    depth_scale_factor = 3.0
else:
    from depth_anything_v2 import (load_depth_model)

    depth_thresholds = [100, 120, 140, 160]
    depth_scale_factor = 0.1

jump = True
min_value = 1e9
max_value = -1e9
use_double_knn_distance = False


def preprocess(path):
    depth_model = load_depth_model()
    min_head_size = 4
    depth_scale_factor = 0.1
    knn_num = 3
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    for folder in ['train', 'test', 'val']:

        images_folder = os.path.join(path, folder, 'images')

        # 图像处理
        images_save_folder = os.path.join(path, folder, 'images_512_2048')
        os.makedirs(images_save_folder, exist_ok=True)

        # 点数据处理
        points_save_folder = os.path.join(path, folder, 'gt_512_2048')
        os.makedirs(points_save_folder, exist_ok=True)

        # 标注文件夹
        annotations_folder = os.path.join(path, folder, 'gt')

        # 点密度图
        density_map_folder = os.path.join(path, folder, 'density_map_512_2048')
        density_map_folder_show = os.path.join(path, folder, 'density_map_512_2048_show')
        os.makedirs(density_map_folder, exist_ok=True)
        os.makedirs(density_map_folder_show, exist_ok=True)

        # 点密度等级图
        density_level_map_folder = os.path.join(path, folder, 'density_level_map_512_2048')
        density_level_map_folder_show = os.path.join(path, folder, 'density_level_map_512_2048_show')
        os.makedirs(density_level_map_folder, exist_ok=True)
        os.makedirs(density_level_map_folder_show, exist_ok=True)

        # 保存depth文件夹
        images_depth_folder = os.path.join(path, folder, 'images_depth_512_2048')
        images_depth_folder_show = os.path.join(path, folder, 'images_depth_512_2048_show')
        os.makedirs(images_depth_folder, exist_ok=True)
        os.makedirs(images_depth_folder_show, exist_ok=True)

        # 人头分割图
        head_split_by_var_folder = os.path.join(path, folder, 'images_head_split_by_var_512_2048')
        head_split_by_depth_folder = os.path.join(path, folder, 'images_head_split_by_depth_512_2048')
        head_split_by_depth_var_folder = os.path.join(path, folder, 'images_head_split_by_depth_var_512_2048')
        os.makedirs(head_split_by_var_folder, exist_ok=True)
        os.makedirs(head_split_by_depth_folder, exist_ok=True)
        os.makedirs(head_split_by_depth_var_folder, exist_ok=True)

        # 保存人头Sizes
        head_size_by_var_folder = os.path.join(path, folder, 'images_head_size_by_var_512_2048')
        head_size_by_depth_folder = os.path.join(path, folder, 'images_head_size_by_depth_512_2048')
        head_size_by_depth_var_folder = os.path.join(path, folder, 'images_head_size_by_depth_var_512_2048')
        os.makedirs(head_size_by_var_folder, exist_ok=True)
        os.makedirs(head_size_by_depth_folder, exist_ok=True)
        os.makedirs(head_size_by_depth_var_folder, exist_ok=True)

        head_split_by_var_folder_show = os.path.join(path, folder, 'images_head_split_by_var_512_2048_show')
        head_split_by_depth_folder_show = os.path.join(path, folder, 'images_head_split_by_depth_512_2048_show')
        head_split_by_depth_var_folder_show = os.path.join(path, folder, 'images_head_split_by_depth_var_512_2048_show')
        os.makedirs(head_split_by_var_folder_show, exist_ok=True)
        os.makedirs(head_split_by_depth_folder_show, exist_ok=True)
        os.makedirs(head_split_by_depth_var_folder_show, exist_ok=True)

        for filename in tqdm(sorted(os.listdir(images_folder))):

            if jump and os.path.exists(os.path.join(head_split_by_depth_var_folder_show, filename)):
                continue

            if not filename.endswith('.jpg'): continue
            image_path = os.path.join(images_folder, filename)
            print(image_path)
            image_pil = Image.open(image_path)
            points = np.loadtxt(os.path.join(annotations_folder, filename.replace('.jpg', '.txt')))
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            if points.shape[1] <= 1:
                points = np.zeros((0, 2))
            points = points[:, 0:2]  # (x, y, ...)

            # resize image 最大2048 最小512
            image_pil, points = resize_image_and_points(image_pil, points, max_size=2048, min_size=512)
            width, height = image_pil.size
            img_save_path = os.path.join(images_save_folder, filename)
            image_pil.save(img_save_path)  # 保存图片

            # 保存点数据
            points_save_path = os.path.join(points_save_folder, filename.replace('.jpg', '.h5'))
            with h5py.File(points_save_path, 'w') as hf:
                hf['points'] = points

            image = np.array(image_pil)

            plot_title = f"JHU Image  {filename.split('.')[0]}"

            # 生成点密度图
            handle_density_map(density_map_folder, density_map_folder_show, filename, image, plot_title, points)

            # 生成点密度等级图
            handle_density_level_map(density_level_map_folder, density_level_map_folder_show, filename, image,
                                     plot_title, points)

            # 深度图
            image_depth_map = handle_depth_map(depth_model, filename, image, images_depth_folder,
                                               images_depth_folder_show,
                                               plot_title, depth_thresholds, use_metric_depth)

            # 根据深度生成人头区域分割图
            points_head_size = handle_depth_split(depth_scale_factor, filename, head_size_by_depth_folder,
                                                  head_split_by_depth_folder, head_split_by_depth_folder_show,
                                                  height, image, image_depth_map, min_head_size, points, width,
                                                  use_metric_depth)

            # 根据距离生成人头区域分割图
            points_average_distances = handle_var_split(filename, head_size_by_var_folder, head_split_by_var_folder,
                                                        head_split_by_var_folder_show, image, knn_num, points,
                                                        use_double_knn_distance)

            # 结合距离和深度生成人头区域分割图
            handle_depth_var_split(filename, head_size_by_depth_var_folder, head_split_by_depth_var_folder,
                                   head_split_by_depth_var_folder_show, image, min_head_size, points,
                                   points_average_distances, points_head_size)


if __name__ == '__main__':
    # root_test = '/mnt/c/Users/lqjun/Desktop'

    root = '/mnt/e/MyDocs/Code/Datasets/jhu_crowd_v2.0'
    preprocess(root)
    print('Process Success!')
    pass
