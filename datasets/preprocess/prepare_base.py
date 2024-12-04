import os

import cv2
import h5py
import numpy as np

from depth_anything_v2_metric import generate_depth_map, max_depth, save_thresholded_depth_map
from utils import select_head_size, generate_mask, draw_rectangles, \
    calculate_nearest_neighbor_distances_double, calculate_average_nearest_neighbor_distance, \
    generate_density_level_map, save_thresholded_density_level_map, generate_gaussian_density_map, save_density_map


def handle_depth_var_split(filename, head_size_by_depth_var_folder, head_split_by_depth_var_folder,
                           head_split_by_depth_var_folder_show, image, min_head_size, points, points_average_distances,
                           points_head_size, save_show=True):
    min_split_head_size = select_head_size(points_head_size, points_average_distances)
    # min_split_head_size = np.minimum(points_average_distances, points_head_size)
    min_split_head_size = np.clip(min_split_head_size, min_head_size, None)
    np.savetxt(os.path.join(head_size_by_depth_var_folder, filename.replace(".jpg", ".txt")),
               min_split_head_size)
    split_head_mask_var_depth = generate_mask(image, points, sizes=min_split_head_size)
    head_split_by_depth_var_path = os.path.join(head_split_by_depth_var_folder, filename)
    cv2.imwrite(head_split_by_depth_var_path, split_head_mask_var_depth * 255)
    if save_show:
        image_with_rectangles = draw_rectangles(image, points, min_split_head_size)
        cv2.imwrite(os.path.join(head_split_by_depth_var_folder_show, filename), image_with_rectangles)  # 绘制矩形框


def handle_var_split(filename, head_size_by_var_folder, head_split_by_var_folder, head_split_by_var_folder_show, image,
                     knn_num, points, use_double_knn_distance, save_show=True):
    points_average_distances = calculate_nearest_neighbor_distances_double(points, knn_num) \
        if use_double_knn_distance else calculate_average_nearest_neighbor_distance(points, knn_num)
    np.savetxt(os.path.join(head_size_by_var_folder, filename.replace(".jpg", ".txt")),
               points_average_distances)  # 保存人头点大小
    split_head_mask_var = generate_mask(image, points, sizes=points_average_distances)
    head_split_by_var_path = os.path.join(head_split_by_var_folder, filename)
    cv2.imwrite(head_split_by_var_path, split_head_mask_var * 255)
    if save_show:
        image_with_rectangles = draw_rectangles(image, points, points_average_distances)
        cv2.imwrite(os.path.join(head_split_by_var_folder_show, filename), image_with_rectangles)  # 绘制矩形框
    return points_average_distances


def handle_depth_split(depth_scale_factor, filename, head_size_by_depth_folder, head_split_by_depth_folder,
                       head_split_by_depth_folder_show, height, image, image_depth_map, min_head_size, points,
                       width, use_metric_depth, save_show=True):
    if len(points) > 0:
        points_round = points.round().astype(int)
        points_round[:, 0] = points_round[:, 0].clip(min=0, max=width - 1)
        points_round[:, 1] = points_round[:, 1].clip(min=0, max=height - 1)
        points_depth = image_depth_map[points_round[:, 1], points_round[:, 0]]
        if use_metric_depth:
            points_depth = 1 / points_depth
        points_head_size = points_depth * depth_scale_factor
        points_head_size = np.clip(points_head_size, min_head_size, None)
    else:
        points_head_size = np.array([])
    np.savetxt(os.path.join(head_size_by_depth_folder, filename.replace(".jpg", ".txt")),
               points_head_size)  # 保存人头点大小
    split_head_mask_depth = generate_mask(image, points, sizes=points_head_size)
    head_split_by_depth_path = os.path.join(head_split_by_depth_folder, filename)
    cv2.imwrite(head_split_by_depth_path, split_head_mask_depth * 255)
    if save_show:
        image_with_rectangles = draw_rectangles(image, points, points_head_size)
        cv2.imwrite(os.path.join(head_split_by_depth_folder_show, filename), image_with_rectangles)  # 绘制矩形框
    return points_head_size


def handle_depth_map(depth_model, filename, image, images_depth_folder, images_depth_folder_show, plot_title,
                     depth_thresholds,
                     use_metric_depth, save_show=True):
    # depth_map = pipe(image)["depth"]
    image_depth_map = generate_depth_map(depth_model, image)
    if use_metric_depth:
        image_depth_map /= max_depth  # 归一化0到1
    depth_map_save_path = os.path.join(images_depth_folder, filename.replace('.jpg', '.h5'))
    with h5py.File(depth_map_save_path, 'w') as hf:
        hf['depth'] = image_depth_map  # 保存深度图
    if save_show:
        depth_map_show_save_path = os.path.join(images_depth_folder_show, filename)
        save_thresholded_depth_map(image_depth_map, depth_map_show_save_path, title=plot_title,
                                   thresholds=depth_thresholds)
    return image_depth_map


def handle_density_level_map(density_level_map_folder, density_level_map_folder_show, filename, image, plot_title,
                             points, save_show=True):
    density_level_map = generate_density_level_map(image, points, window_size=128, n_jobs=4)
    density_level_map_save_path = os.path.join(density_level_map_folder, filename.replace('.jpg', '.h5'))
    with h5py.File(density_level_map_save_path, 'w') as hf:
        hf['density_level'] = density_level_map  # 生成点密度等级图
    if save_show:
        save_thresholded_density_level_map(density_level_map, os.path.join(density_level_map_folder_show, filename),
                                           title=plot_title, thresholds=[10, 15, 20, 30])


def handle_density_map(density_map_folder, density_map_folder_show, filename, image, plot_title, points,
                       save_show=True):
    # density_map = generate_point_density_map(image, points, window_size=32)
    density_map = generate_gaussian_density_map(image, points, sigma=4)
    density_map_save_path = os.path.join(density_map_folder, filename.replace('.jpg', '.h5'))
    with h5py.File(density_map_save_path, 'w') as hf:
        hf['density'] = density_map  # 生成点密度图
    if save_show:
        save_density_map(density_map, os.path.join(density_map_folder_show, filename), title=plot_title)
    # save_thresholded_density_map(density_map, os.path.join(density_map_folder_show, filename), title=plot_title, thresholds=[0.15, 0.02, 0.25, 0.03])
