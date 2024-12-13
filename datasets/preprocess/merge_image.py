from PIL import Image, ImageDraw, ImageFont


def add_title_to_image(image, title, position='top', bg_color="#f0f0f0", font_size=50):
    """
    在图像上添加标题。

    :param image: 要添加标题的图像
    :param title: 标题文本
    :param position: 标题位置，'top' 或 'bottom'
    :param bg_color: 背景颜色
    :param font_size: 字体大小
    :return: 添加标题后的图像
    """
    # 获取图像的宽度和高度
    width, height = image.size

    # 使用一个支持 Unicode 的字体（例如 DejaVuSans.ttf）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
    except IOError:
        font = ImageFont.load_default()  # 如果找不到字体，则使用默认字体（但是支持有限）

    # 使用 textbbox 来计算文本的边界框
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), title, font=font)  # 获取文本边界框
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    if position == 'bottom':
        new_height = height + text_height + 20  # 留出一些空白
        new_image = Image.new('RGB', (width, new_height), bg_color)
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        draw.text(((width - text_width) / 2, height + 10), title, font=font, fill="black")
    elif position == 'top':
        new_height = height + text_height + 30  # 留出一些空白
        new_image = Image.new('RGB', (width, new_height), bg_color)
        draw = ImageDraw.Draw(new_image)
        draw.text(((width - text_width) / 2, 10), title, font=font, fill="black")
        new_image.paste(image, (0, text_height + 30))
    else:
        raise ValueError("position must be 'top' or 'bottom'")

    return new_image


def stack_images(image_paths, image_titles: list = None, direction='vertical', spacing=10, bg_color="#f0f0f0"):
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

    # 如果有标题，则为每张图像添加标题
    if image_titles:
        first_image = add_title_to_image(first_image, image_titles[0], position='top', bg_color=bg_color)
        images[0] = first_image

    # 计算总宽度或高度
    total_width = width if direction == 'horizontal' else 0
    total_height = height if direction == 'vertical' else 0

    for i, path in enumerate(image_paths[1:], 1):
        img = Image.open(path)
        if image_titles:
            img = add_title_to_image(img, image_titles[i], position='top', bg_color=bg_color)
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


if __name__ == '__main__':
    # 示例用法
    # image_paths = [
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_depth_show/IMG_2.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_depth_show/IMG_85.jpg',
    # ]  # 图像文件路径列表
    # stacked_image = stack_images(image_paths, direction='vertical', spacing=20)
    # stacked_image.save("depth.jpg")

    image_paths = [
        '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_var_show/IMG_100.jpg',
        '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_show/IMG_100.jpg',
        '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_var_show/IMG_100.jpg',
    ]
    image_titles = ['使用KNN距离估计尺寸', '使用深度值估计尺寸', '结合KNN距离和深度值估计尺寸']
    stacked_image = stack_images(image_paths, image_titles=image_titles, direction='horizontal', spacing=20,
                                 bg_color="#ffffff")
    stacked_image.save("head_size_comparison.jpg")

    print("merge sucess!")
