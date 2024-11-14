import matplotlib
import matplotlib.pyplot as plt
import torch

from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# 设置 matplotlib 使用 Agg 后端
matplotlib.use('Agg')

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl'  # or 'vits', 'vitb'
dataset = 'vkitti'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 80  # 20 for indoor model, 80 for outdoor model


def load_depth_model():
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(
        torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth',
                   map_location='cpu'))
    model = model.to(DEVICE).eval()

    return model


def generate_depth_map(model, raw_img):
    depth_map = model.infer_image(raw_img)  # HxW raw depth map in numpy
    return depth_map


def save_depth_map(depth_map, save_path: str, title: str):
    # 可视化深度图
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap='viridis', vmin=0.0, vmax=1.0)  # 使用viridis颜色映射
    plt.colorbar(label='Depth')  # 添加颜色条
    plt.title(title)
    plt.axis('off')  # 关闭坐标轴

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_thresholded_depth_map(depth_map, save_path: str, title: str, thresholds):
    """
    根据给定的阈值绘制分割图，最后将原始图像和分割图放在一行上显示。
    """

    # 创建一个1行(len(thresholds) + 1)列的子图布局
    fig, axs = plt.subplots(1, len(thresholds) + 1, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f0f0')  # 设置整个图形的背景颜色

    # 显示原始图像
    im = axs[0].imshow(depth_map, cmap='viridis', vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Depth', labelpad=5)
    axs[0].set_title(title)
    axs[0].axis('off')

    # 对每个阈值进行分割并显示
    for i, threshold in enumerate(thresholds):
        # 使用阈值进行分割
        binary_image = depth_map > threshold

        # 显示分割后的图像
        axs[i + 1].imshow(binary_image, cmap='gray')
        axs[i + 1].set_title(f'Threshold: {threshold}')
        axs[i + 1].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存深度图到文件
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
