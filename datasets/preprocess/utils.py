import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# 设置 matplotlib 使用 Agg 后端
matplotlib.use('Agg')


def resize_image_and_points_by_min_edge(img: Image.Image, points: ndarray, min_edge_max_size: int = 2048):
    # 读取图片
    width, height = img.size

    img_min_size = min(width, height)

    # 计算缩放比例
    scale_factor = 1.0
    if img_min_size > min_edge_max_size:
        scale_factor = min_edge_max_size / img_min_size

    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 调整图片大小
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # 调整点
        points = points * scale_factor
    return img, points


def resize_image_and_points(img: Image.Image, points: ndarray, max_size: int = 2048, min_size: int = 512):
    # 读取图片
    width, height = img.size

    img_max_size = max(width, height)
    img_min_size = min(width, height)

    # 计算缩放比例
    scale_factor = 1.0
    if img_max_size > max_size:
        scale_factor = max_size / img_max_size
    if img_min_size < min_size:
        scale_factor = min_size / img_min_size

    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 调整图片大小
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # 调整点
        points = points * scale_factor
    return img, points


def save_to_h5(output_path, image, points):
    # 将图片转换为 numpy 数组
    image_array = np.array(image)

    # 创建 HDF5 文件
    with h5py.File(output_path, 'w') as hf:
        # 保存图片
        hf.create_dataset('image', data=image_array)

        # 保存 points
        points_array = np.array(points)
        hf.create_dataset('points', data=points_array)


def save_image(path, image_arr, boxes):
    # 遍历每个目标框并绘制到图像上
    for box in boxes:
        x, y, w, h = box
        x1, y1 = x, y  # 计算左上角坐标
        x2, y2 = x + w, y + h  # 计算右下角坐标
        cv2.rectangle(image_arr, (x1.round().astype(int), y1.round().astype(int)),
                      (x2.round().astype(int), y2.round().astype(int)), (0, 255, 0), 1)  # 绘制矩形框，颜色为绿色，线宽为2
    # 保存图像
    cv2.imwrite(path, image_arr)


def calculate_average_nearest_neighbor_distance(points, k=3, default=16.0):
    if len(points) == 0:
        return np.array([])
    if len(points) == 1:
        return np.array([default])
    if len(points) <= k:
        k = len(points) - 1

    # 计算点之间的距离矩阵
    distance_matrix = cdist(points, points)

    # 对每个点的距离矩阵进行排序，并获取前k个最近邻的距离
    nearest_neighbors_distances = np.sort(distance_matrix, axis=1)[:, 1:k + 1]

    # 计算每个点的最近邻的平均距离
    average_distances = np.mean(nearest_neighbors_distances, axis=1)

    return 0.5 * average_distances


def calculate_nearest_neighbor_distances_double(points, k=3, default=16.0):
    """
    计算每个点到最近邻的k个点的平均距离，然后再次选择每个点的最近邻的k个点，
    从平均距离中选取这k个点对应的值，计算对应的平均值。

    参数:
    points (np.ndarray): 形状为 (n, 2) 的数组，表示 n 个点数据。
    k (int): 最近邻点的数量。

    返回:
    tuple: 包含两个数组，第一个数组是每个点到最近邻的k个点的平均距离，
           第二个数组是再次选择每个点的最近邻的k个点对应的平均值。
    """
    if len(points) == 0:
        return np.array([])
    if len(points) == 1:
        return np.array([default])
    if len(points) <= k:
        k = len(points) - 1

    # 构建KDTree
    tree = KDTree(points)

    # 查询每个点的最近邻的k+1个点（包括自身）
    distances, indices = tree.query(points, k=k + 1)

    # 计算每个点到最近邻的k个点的平均距离（排除自身）
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # 再次选择每个点的最近邻的k个点（排除自身）
    nearest_indices = indices[:, 1:]

    # 从avg_distances中选取这k个点对应的值，计算对应的平均值
    nearest_avg_distances = avg_distances[nearest_indices]
    final_avg_distances = np.mean(nearest_avg_distances, axis=1)

    return 0.5 * final_avg_distances


def generate_mask(image, points, sizes):
    # 获取图像的形状
    H, W = image.shape[0:2]

    # 初始化mask，所有值为0
    mask = np.zeros((H, W), dtype=np.uint8)

    # 遍历每个点
    for (x, y), size in zip(points, sizes):
        # 计算点的范围
        x_min = int(max(0, x - size // 2))
        x_max = int(min(W, x + size // 2 + 1))
        y_min = int(max(0, y - size // 2))
        y_max = int(min(H, y + size // 2 + 1))

        # 在mask中设置点的范围为1
        mask[y_min:y_max, x_min:x_max] = 1

    return mask


def generate_circular_mask(image, points, sizes):
    # 获取图像的形状
    H, W = image.shape[0:2]

    # 初始化mask，所有值为0
    mask = np.zeros((H, W), dtype=np.uint8)

    # 遍历每个点
    for (x, y), size in zip(points, sizes):
        # 计算圆的半径
        radius = size // 2

        # 生成圆形mask
        y_grid, x_grid = np.ogrid[:H, :W]
        circle_mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2

        # 在mask中设置圆的范围为1
        mask[circle_mask] = 1

    return mask


def draw_rectangles(image, points, sizes):
    # 复制图像以避免直接修改原始图像
    image_with_rectangles = image.copy()

    # 遍历每个点
    for (x, y), size in zip(points, sizes):
        # 计算矩形框的左上角和右下角坐标
        half_size = size // 2
        x_min = int(max(0, x - half_size))
        y_min = int(max(0, y - half_size))
        x_max = int(min(image.shape[1], x + half_size))
        y_max = int(min(image.shape[0], y + half_size))

        # 绘制矩形框
        cv2.rectangle(image_with_rectangles, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return image_with_rectangles


# 假设image是一个二维numpy数组，points是一个形状为(N, 2)的数组，包含点的(x, y)坐标
def generate_gaussian_density_map(image, points, sigma=4):
    # 获取图像的尺寸
    H, W = image.shape[:2]

    # 创建一个与图像大小相同的密度图
    density_map = np.zeros((H, W))

    # 遍历每个点，将其位置在密度图上标记为1
    for point in points:
        x = min(max(int(point[0]), 0), density_map.shape[1] - 1)
        y = min(max(int(point[1]), 0), density_map.shape[0] - 1)
        density_map[y, x] += 1

    # 使用高斯滤波器对密度图进行平滑处理
    density_map = gaussian_filter(density_map, sigma=sigma)

    # 归一化密度图
    density_map /= np.sum(density_map)
    density_map *= len(points)

    return density_map


def generate_point_density_map(image, points, window_size=32):
    # 获取图像的形状
    H, W = image.shape[:2]

    # 初始化密度图，所有值为0
    density_map = np.zeros((H, W), dtype=np.float32)

    # 计算每个点的密度贡献
    for x, y in points:
        # 计算窗口的左上角和右下角坐标
        x_min = int(max(0, x - window_size // 2))
        y_min = int(max(0, y - window_size // 2))
        x_max = int(min(W, x + window_size // 2))
        y_max = int(min(H, y + window_size // 2))

        # 在密度图中增加点的贡献
        density_map[y_min:y_max, x_min:x_max] += 1

    return density_map


def save_density_map(density_map, save_path: str, title: str):
    # 可视化深度图
    plt.figure(figsize=(8, 6))
    plt.imshow(density_map, cmap='viridis', vmin=0.0, vmax=1.0)  # 使用viridis颜色映射
    plt.colorbar(label='Density')  # 添加颜色条
    plt.title(title)
    plt.axis('off')  # 关闭坐标轴

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_thresholded_density_map(density_map, save_path: str, title: str, thresholds):
    """
    根据给定的阈值绘制分割图，最后将原始图像和分割图放在一行上显示。
    """

    # 创建一个1行(len(thresholds) + 1)列的子图布局
    fig, axs = plt.subplots(1, len(thresholds) + 1, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f0f0')  # 设置整个图形的背景颜色

    # 显示原始图像
    im = axs[0].imshow(density_map, cmap='viridis')  # , vmin=0.0, vmax=0.04
    cbar = fig.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Density', labelpad=5)
    axs[0].set_title(title)
    axs[0].axis('off')

    # 对每个阈值进行分割并显示
    for i, threshold in enumerate(thresholds):
        # 使用阈值进行分割
        binary_image = density_map > threshold

        # 显示分割后的图像
        axs[i + 1].imshow(binary_image, cmap='gray')
        axs[i + 1].set_title(f'Threshold: {threshold}')
        axs[i + 1].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_density_level_map(image, points, window_size=128, n_jobs=-1):
    height, width = image.shape[0:2]

    def process_pixel(y, x, points, window_size):
        y_min = max(0, y - window_size // 2)
        y_max = min(height, y + window_size // 2 + 1)
        x_min = max(0, x - window_size // 2)
        x_max = min(width, x + window_size // 2 + 1)

        if len(points) == 0:
            return 0.5 * np.sqrt(2) * window_size

        local_points = points[
            (points[:, 0] >= x_min) & (points[:, 0] < x_max) & (points[:, 1] >= y_min) & (points[:, 1] < y_max)]

        if len(local_points) == 0 or len(local_points) == 1:
            return 0.5 * np.sqrt(2) * window_size

        distances = np.linalg.norm(local_points - np.array([x, y]), axis=1)

        if len(distances) < 3:
            avg_distance = np.mean(distances)
        else:
            avg_distance = np.mean(np.sort(distances)[:3])

        return avg_distance

    # 使用并行计算加速
    density_level_map = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel)(y, x, points, window_size)
        for y in range(height)
        for x in range(width)
    )

    return np.array(density_level_map).reshape(height, width)


def save_thresholded_density_level_map(density_level_map, save_path: str, title: str, thresholds):
    """
    根据给定的阈值绘制分割图，最后将原始图像和分割图放在一行上显示。
    """

    # 创建一个1行(len(thresholds) + 1)列的子图布局
    fig, axs = plt.subplots(1, len(thresholds) + 1, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f0f0')  # 设置整个图形的背景颜色

    # 显示原始图像
    im = axs[0].imshow(density_level_map, cmap='viridis', vmin=0.0, vmax=200)  #
    cbar = fig.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Density Level', labelpad=5)
    axs[0].set_title(title)
    axs[0].axis('off')

    # 对每个阈值进行分割并显示
    for i, threshold in enumerate(thresholds):
        # 使用阈值进行分割
        binary_image = density_level_map > threshold

        # 显示分割后的图像
        axs[i + 1].imshow(binary_image, cmap='gray')
        axs[i + 1].set_title(f'Threshold: {threshold}')
        axs[i + 1].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# def get_distribute_by_head_size(head_sizes):
#     """获取分布情况"""
#     # 平均深度
#     max_head_size = max(head_sizes)
#     min_head_size = min(head_sizes)
#     avg_head_size = statistics.mean(head_sizes)
#
#     distribute_type = None
#     if (max_head_size - min_head_size) / avg_head_size < 0.5:
#         distribute_type = 0  # 分布变化不太大的情况
#     else:
#         m_value = (max_head_size + min_head_size) / 2
#         if abs(avg_head_size - m_value) < (max_head_size - min_head_size) * 0.1:
#             distribute_type = 1 # 大尺寸和小尺寸人头数量差不多
#         elif avg_head_size > m_value:
#             distribute_type = 2  # 大的尺寸多一点
#         else:
#             distribute_type = 3  # 小的尺寸多一点
#
#     return distribute_type

def select_head_size(depth_head_size, var_head_size):
    if len(depth_head_size) <= 2:  # 点非常少的情况
        return depth_head_size

    # depth_distribute_type = get_distribute_by_head_size(depth_head_size)
    # var_distribute_type = get_distribute_by_head_size(var_head_size)

    result = []
    for d_size, a_size in zip(depth_head_size, var_head_size):
        size = -1
        if d_size > a_size * 1.5:
            size = a_size * 1.5
        elif d_size < a_size / 1.5:
            size = a_size / 1.5
            # match var_distribute_type:
            #     case 0: size = a_size / 1.5
            #     case 1: size = d_size * 1.5
            #     case 2: size = a_size
            #     case 3: size = d_size
        else:
            size = d_size

        assert size != -1
        result.append(size)
    return np.array(result)


def exception_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An exception occurred: {e}")
            return "Error: Function execution failed."

    return wrapper
