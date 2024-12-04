import os

import h5py
import numpy as np
# from transformers import pipeline
from PIL import Image
from joblib import Parallel, delayed
from scipy.io import loadmat
from tqdm import tqdm

from prepare_base import handle_depth_var_split, handle_var_split, handle_depth_split, \
    handle_depth_map
from utils import resize_image_and_points, resize_image_and_points_by_min_edge, split_image_and_points

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
save_show = False
min_value = 1e9
max_value = -1e9
use_double_knn_distance = False

n_jobs = 1
split_size = 512
scale_min_edge = True
if scale_min_edge:
    min_edge_max_size = 2048
    scale_info = f'min_edge_{min_edge_max_size}_split_{split_size}'
else:
    img_min_size = 0
    img_max_size = 2048
    scale_info = f'{img_min_size}_{img_max_size}_split'


def preprocess(path):
    depth_model = load_depth_model()
    min_head_size = 4
    depth_scale_factor = 0.1
    knn_num = 3
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    for folder in ['Train', 'Test']:
        # for folder in ['Test']:

        # 图片文件夹
        images_folder = os.path.join(path, folder)

        # 标注文件夹
        annotations_folder = images_folder

        save_folder = os.path.join('processed', scale_info, folder)
        os.makedirs(save_folder, exist_ok=True)

        # 图像处理
        images_save_folder = os.path.join(path, save_folder, 'images')
        os.makedirs(images_save_folder, exist_ok=True)

        # 点数据处理
        points_save_folder = os.path.join(path, save_folder, 'gt')
        os.makedirs(points_save_folder, exist_ok=True)

        # 点密度图
        density_map_folder = os.path.join(path, save_folder, 'density_map')
        density_map_folder_show = os.path.join(path, save_folder, 'density_map_show')
        os.makedirs(density_map_folder, exist_ok=True)
        os.makedirs(density_map_folder_show, exist_ok=True)

        # 点密度等级图
        density_level_map_folder = os.path.join(path, save_folder, 'density_level_map')
        density_level_map_folder_show = os.path.join(path, save_folder, 'density_level_map_show')
        os.makedirs(density_level_map_folder, exist_ok=True)
        os.makedirs(density_level_map_folder_show, exist_ok=True)

        # 保存depth文件夹
        images_depth_folder = os.path.join(path, save_folder, 'images_depth')
        images_depth_folder_show = os.path.join(path, save_folder, 'images_depth_show')
        os.makedirs(images_depth_folder, exist_ok=True)
        os.makedirs(images_depth_folder_show, exist_ok=True)

        # 人头分割图
        head_split_by_var_folder = os.path.join(path, save_folder, 'images_head_split_by_var')
        head_split_by_depth_folder = os.path.join(path, save_folder, 'images_head_split_by_depth')
        head_split_by_depth_var_folder = os.path.join(path, save_folder, 'images_head_split_by_depth_var')
        os.makedirs(head_split_by_var_folder, exist_ok=True)
        os.makedirs(head_split_by_depth_folder, exist_ok=True)
        os.makedirs(head_split_by_depth_var_folder, exist_ok=True)

        # 保存人头Sizes
        head_size_by_var_folder = os.path.join(path, save_folder, 'images_head_size_by_var')
        head_size_by_depth_folder = os.path.join(path, save_folder, 'images_head_size_by_depth')
        head_size_by_depth_var_folder = os.path.join(path, save_folder, 'images_head_size_by_depth_var')
        os.makedirs(head_size_by_var_folder, exist_ok=True)
        os.makedirs(head_size_by_depth_folder, exist_ok=True)
        os.makedirs(head_size_by_depth_var_folder, exist_ok=True)

        head_split_by_var_folder_show = os.path.join(path, save_folder, 'images_head_split_by_var_show')
        head_split_by_depth_folder_show = os.path.join(path, save_folder, 'images_head_split_by_depth_show')
        head_split_by_depth_var_folder_show = os.path.join(path, save_folder, 'images_head_split_by_depth_var_show')
        os.makedirs(head_split_by_var_folder_show, exist_ok=True)
        os.makedirs(head_split_by_depth_folder_show, exist_ok=True)
        os.makedirs(head_split_by_depth_var_folder_show, exist_ok=True)

        for filename in tqdm(sorted(os.listdir(images_folder))):

            # if jump and os.path.exists(os.path.join(head_split_by_depth_var_folder_show, filename)):
            #     continue
            # filename = "IMG_108.jpg" # 问题图片
            if not filename.endswith('.jpg'): continue
            image_path = os.path.join(images_folder, filename)
            print(image_path)
            image_pil = Image.open(image_path)
            points = \
                loadmat(os.path.join(annotations_folder, filename.replace('.jpg', '_ann.mat')))['annPoints']
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            if points.shape[1] <= 1:
                points = np.zeros((0, 2))
            points = points[:, 0:2]  # (x, y, ...)

            if scale_min_edge:
                image_pil, points = resize_image_and_points_by_min_edge(image_pil, points,
                                                                        min_edge_max_size=min_edge_max_size)
            else:
                image_pil, points = resize_image_and_points(image_pil, points, max_size=img_max_size,
                                                            min_size=img_min_size)

            # 分割成块
            expand_ratio = 0.25
            overlap_ratio = 0.25 if folder == 'Train' else 0.0
            img_blocks, points_in_blocks = split_image_and_points(image_pil, points, split_size=split_size,
                                                                  overlap_ratio=overlap_ratio,
                                                                  expand_ratio=expand_ratio)

            def process_core(idx, img_block, points_block):
                points_block = np.array(points_block)
                if points_block.ndim == 1:
                    points_block = np.expand_dims(points_block, axis=0)
                if points_block.shape[1] <= 1:
                    points_block = np.zeros((0, 2))

                file_name_without_ext = os.path.splitext(filename)[0]
                block_filename = f"{file_name_without_ext}_{idx:02}.jpg"

                width, height = img_block.size
                image = np.array(img_block)
                img_save_path = os.path.join(images_save_folder, block_filename)
                img_block.save(img_save_path)  # 保存图片

                # 保存点数据
                points_save_path = os.path.join(points_save_folder, block_filename.replace('.jpg', '.h5'))
                with h5py.File(points_save_path, 'w') as hf:
                    hf['points'] = points_block

                plot_title = f"SHA Image {block_filename.split('.')[0]}"

                # 生成点密度图
                # handle_density_map(density_map_folder, density_map_folder_show, filename, image, plot_title, points, save_show=save_show)

                # 生成点密度等级图
                # handle_density_level_map(density_level_map_folder, density_level_map_folder_show, filename, image,
                #                          plot_title, points, save_show=save_show)

                # 深度图
                image_depth_map = handle_depth_map(depth_model, block_filename, image, images_depth_folder,
                                                   images_depth_folder_show,
                                                   plot_title, depth_thresholds, use_metric_depth, save_show=save_show)

                # 根据深度生成人头区域分割图
                points_head_size = handle_depth_split(depth_scale_factor, block_filename, head_size_by_depth_folder,
                                                      head_split_by_depth_folder, head_split_by_depth_folder_show,
                                                      height, image, image_depth_map, min_head_size, points_block,
                                                      width,
                                                      use_metric_depth, save_show=save_show)

                # 根据距离生成人头区域分割图
                points_average_distances = handle_var_split(block_filename, head_size_by_var_folder,
                                                            head_split_by_var_folder,
                                                            head_split_by_var_folder_show, image, knn_num, points_block,
                                                            use_double_knn_distance, save_show=save_show)

                # 结合距离和深度生成人头区域分割图
                handle_depth_var_split(block_filename, head_size_by_depth_var_folder, head_split_by_depth_var_folder,
                                       head_split_by_depth_var_folder_show, image, min_head_size, points_block,
                                       points_average_distances, points_head_size, save_show=save_show)

            # 使用并行计算加速
            Parallel(n_jobs=n_jobs)(
                delayed(process_core)(idx, img_block, points_block)
                for idx, (img_block, points_block) in enumerate(zip(img_blocks, points_in_blocks))
            )
            # break


def save_image_points_num(path):
    for folder in ['Train', 'Test']:

        # 图片文件夹
        images_folder = os.path.join(path, folder)

        # 标注文件夹
        annotations_folder = images_folder

        save_folder = os.path.join('processed', scale_info, folder)
        os.makedirs(save_folder, exist_ok=True)

        image_points_num_dict = dict()
        image_points_num_save_file = os.path.join(path, save_folder, 'image_points_num.npy')

        for filename in tqdm(sorted(os.listdir(images_folder))):
            img_name_without_ext = os.path.splitext(filename)[0]
            points = loadmat(os.path.join(annotations_folder, filename.replace('.jpg', '_ann.mat')))['annPoints']
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            if points.shape[1] <= 1:
                points = np.zeros((0, 2))
            points = points[:, 0:2]  # (x, y, ...)

            image_points_num_dict[img_name_without_ext] = points.shape[0]
            # break

        np.save(image_points_num_save_file, image_points_num_dict)


if __name__ == '__main__':
    # root_test = '/mnt/c/Users/lqjun/Desktop'

    root = '../../data/UCF-QNRF_ECCV18'
    preprocess(root)
    save_image_points_num(root)
    print('Process Success!')
    pass
