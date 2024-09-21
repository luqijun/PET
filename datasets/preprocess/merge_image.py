from PIL import Image

def stack_images(image_paths, direction='vertical', spacing=10, bg_color="#f0f0f0"):
    """
    将多张图像堆叠在一起，可以选择水平或垂直堆叠，并设置间隔。

    :param image_paths: 图像文件路径列表
    :param direction: 堆叠方向，'horizontal' 或 'vertical'
    :param spacing: 图像之间的间隔
    :return: 堆叠后的图像
    """
    # 读取第一张图像以获取尺寸
    first_image = Image.open(image_paths[0])
    width, height = first_image.size
    images = [first_image]

    # 计算总宽度或高度
    total_width = width if direction == 'horizontal' else 0
    total_height = height if direction == 'vertical' else 0

    for path in image_paths[1:]:
        img = Image.open(path)
        images.append(img)
        if direction == 'horizontal':
            total_width += img.width + spacing
            total_height = max(total_height, img.height)
        else:
            total_width = max(total_width, img.width)
            total_height += img.height + spacing

    # 创建一个新的空白图像
    stacked_image = Image.new('RGB', (total_width, total_height), bg_color)

    # 将图像堆叠在一起
    x_offset = 0
    y_offset = 0
    for img in images:
        if direction == 'horizontal':
            stacked_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
        else:
            stacked_image.paste(img, (0, y_offset))
            y_offset += img.height + spacing

    return stacked_image

# 示例用法
image_paths = [
    '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet/density_level_map_show/IMG_2.jpg',
    '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet/density_level_map_show/IMG_85.jpg',
]  # 图像文件路径列表
stacked_image = stack_images(image_paths, direction='vertical', spacing=20)
stacked_image.save("merge.jpg")