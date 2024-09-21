import os
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
# from transformers import pipeline
from PIL import Image
import scipy.ndimage as ndimage
import cv2
from tqdm import tqdm
import h5py
from depth_anything_v2 import load_depth_model, generate_depth_map, save_depth_map, save_thresholded_depth_map
from utils import (resize_image_and_points, calculate_nearest_neighbor_distances_double,
                   generate_mask, generate_circular_mask, draw_rectangles,
                   generate_gaussian_density_map, generate_point_density_map, save_density_map, save_thresholded_density_map,
                   generate_density_level_map, save_thresholded_density_level_map,
                   select_head_size, exception_wrapper)

min_value = 1e9
max_value = -1e9

jump = True

def preprocess(path):

    load_depth_model()
    min_head_size = 4
    depth_scale_factor = 0.1
    knn_num = 3
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    for folder in ['train', 'test', 'val']:

        images_folder = os.path.join(path, folder, 'images')

        # 标注文件夹
        annotations_folder = os.path.join(path, folder, 'gt')

        # images process save folder
        images_save_folder = os.path.join(path, folder, 'images_512_2048')
        os.makedirs(images_save_folder, exist_ok=True)

        points_save_folder = os.path.join(path, folder, 'gt_512_2048')
        os.makedirs(points_save_folder, exist_ok=True)

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
            image_pil.save(img_save_path) # 保存图片

            # 保存点数据
            points_save_path = os.path.join(points_save_folder, filename.replace('.jpg', '.h5'))
            with h5py.File(points_save_path, 'w') as hf:
                hf['points'] = points

            image = np.array(image_pil)

            plot_title = f"JHU Image  {filename.split('.')[0]}"

            # 生成点密度图
            # density_map = generate_point_density_map(image, points, window_size=32)
            density_map = generate_gaussian_density_map(image, points, sigma=4)
            density_map_save_path = os.path.join(density_map_folder, filename.replace('.jpg', '.h5'))
            with h5py.File(density_map_save_path, 'w') as hf:
                hf['density'] = density_map # 生成点密度图
            save_density_map(density_map, os.path.join(density_map_folder_show, filename), title=plot_title)
            # save_thresholded_density_map(density_map, os.path.join(density_map_folder_show, filename), title=plot_title, thresholds=[0.15, 0.02, 0.25, 0.03])

            # 生成点密度等级图
            density_level_map = generate_density_level_map(image, points, window_size=128, n_jobs=4)
            density_level_map_save_path = os.path.join(density_level_map_folder, filename.replace('.jpg', '.h5'))
            with h5py.File(density_level_map_save_path, 'w') as hf:
                hf['density_level'] = density_level_map # 生成点密度等级图
            save_thresholded_density_level_map(density_level_map, os.path.join(density_level_map_folder_show, filename), title=plot_title, thresholds=[10, 15, 20, 30])

            # 深度图
            # depth_map = pipe(image)["depth"]
            image_depth_map = generate_depth_map(image)
            depth_map_save_path = os.path.join(images_depth_folder, filename.replace('.jpg', '.h5'))
            with h5py.File(depth_map_save_path, 'w') as hf:
                hf['depth'] = image_depth_map # 保存深度图
            depth_map_show_save_path = os.path.join(images_depth_folder_show, filename)
            # save_depth_map(image_depth_map, depth_map_show_save_path, title=plot_title) # 深度图可视化保存
            save_thresholded_depth_map(image_depth_map, depth_map_show_save_path, title=plot_title, thresholds=[100, 120, 140, 160])

            # 根据距离生成人头区域分割图
            points_average_distances = calculate_nearest_neighbor_distances_double(points, knn_num)
            split_head_mask_var = generate_mask(image, points, sizes=points_average_distances)
            head_split_by_var_path = os.path.join(head_split_by_var_folder, filename)
            cv2.imwrite(head_split_by_var_path, split_head_mask_var * 255)
            image_with_rectangles = draw_rectangles(image, points, points_average_distances)
            cv2.imwrite(os.path.join(head_split_by_var_folder_show, filename), image_with_rectangles) # 绘制矩形框

            # 根据深度生成人头区域分割图
            if len(points) > 0:
                points_round = points.round().astype(int)
                points_round[:, 0] = points_round[:, 0].clip(min=0, max=width - 1)
                points_round[:, 1] = points_round[:, 1].clip(min=0, max=height - 1)
                points_depth = image_depth_map[points_round[:, 1], points_round[:, 0]]
                points_head_size = points_depth * depth_scale_factor
                points_head_size = np.clip(points_head_size, min_head_size, None)
            else:
                points_head_size = np.array([])
            split_head_mask_depth = generate_mask(image, points, sizes=points_head_size)
            head_split_by_depth_path = os.path.join(head_split_by_depth_folder, filename)
            cv2.imwrite(head_split_by_depth_path, split_head_mask_depth * 255)
            image_with_rectangles = draw_rectangles(image, points, points_head_size)
            cv2.imwrite(os.path.join(head_split_by_depth_folder_show, filename), image_with_rectangles) # 绘制矩形框

            # 结合距离和深度生成人头区域分割图
            min_split_head_size = select_head_size(points_head_size, points_average_distances)
            # min_split_head_size = np.minimum(points_average_distances, points_head_size)
            min_split_head_size = np.clip(min_split_head_size, min_head_size, None)
            split_head_mask_var_depth = generate_mask(image, points, sizes=min_split_head_size)
            head_split_by_depth_var_path = os.path.join(head_split_by_depth_var_folder, filename)
            cv2.imwrite(head_split_by_depth_var_path, split_head_mask_var_depth * 255)
            image_with_rectangles = draw_rectangles(image, points, min_split_head_size)
            cv2.imwrite(os.path.join(head_split_by_depth_var_folder_show, filename), image_with_rectangles)  # 绘制矩形框

            # depth_img = np.array(depth)


if __name__ == '__main__':
    # root_test = '/mnt/c/Users/lqjun/Desktop'

    root = '/mnt/e/MyDocs/Code/Datasets/jhu_crowd_v2.0'
    preprocess(root)
    print('Process Success!')
    pass
