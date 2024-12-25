from PIL import Image, ImageDraw, ImageFont


def add_title_to_image(image, title, position='top', bg_color="#f0f0f0", spacing=10, font_size=50,
                       fix_text_height=None):
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
    if fix_text_height is not None:
        text_height = fix_text_height

    if position == 'bottom':
        new_height = height + text_height + spacing  # 留出一些空白
        new_image = Image.new('RGB', (width, new_height), bg_color)
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        draw.text(((width - text_width) / 2, height + spacing), title, font=font, fill="black")
    elif position == 'top':
        new_height = height + text_height + spacing * 2  # 留出一些空白
        new_image = Image.new('RGB', (width, new_height), bg_color)
        draw = ImageDraw.Draw(new_image)
        draw.text(((width - text_width) / 2, spacing), title, font=font, fill="black")
        new_image.paste(image, (0, text_height + spacing * 2))
    else:
        raise ValueError("position must be 'top' or 'bottom'")

    return new_image


def stack_images(image_paths, *, image_titles: list = None, direction='vertical', spacing=10, bg_color="#f0f0f0",
                 font_size=50,
                 cut_range=None):
    """
    将多张图像堆叠在一起，可以选择水平或垂直堆叠，并设置间隔。

    :param image_paths: 图像文件路径列表
    :param direction: 堆叠方向，'horizontal' 或 'vertical'
    :param spacing: 图像之间的间隔
    :param cut_range: 裁剪范围，数组格式[(x1,y1), (x2,y2)]，代表左上角坐标点和右下角坐标点
    :return: 堆叠后的图像
    """
    # 读取第一张图像以获取尺寸
    first_image = Image.open(image_paths[0])
    # 裁剪
    if cut_range is not None:
        first_image = cut_image(first_image, cut_range)

    width, height = first_image.size
    images = [first_image]

    # 如果有标题，则为每张图像添加标题
    fix_text_height = 16
    if image_titles:
        first_image = add_title_to_image(first_image, image_titles[0], position='top', bg_color=bg_color,
                                         spacing=spacing,
                                         fix_text_height=fix_text_height,
                                         font_size=font_size)
        images[0] = first_image

    # 计算总宽度或高度
    total_width = width if direction == 'horizontal' else 0
    total_height = height if direction == 'vertical' else 0

    for i, path in enumerate(image_paths[1:], 1):
        img = Image.open(path)
        if cut_range is not None:
            img = cut_image(img, cut_range)
        if image_titles:
            img = add_title_to_image(img, image_titles[i], position='top', bg_color=bg_color,
                                     fix_text_height=fix_text_height, font_size=font_size)
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


def cut_image(image, cut_range):
    """
    按指定的裁剪范围裁剪图像。

    :param image: PIL 图像对象
    :param cut_range: 裁剪范围，格式为 [(x1, y1), (x2, y2)]，代表左上角坐标点和右下角坐标点
    :return: 被裁剪后的图像
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image 参数必须是一个 PIL 图像对象")

    if cut_range is None or len(cut_range) != 2 or not all(len(coord) == 2 for coord in cut_range):
        raise ValueError("cut_range 必须是形如 [(x1, y1), (x2, y2)] 的数组")

    x1, y1 = cut_range[0]
    x2, y2 = cut_range[1]

    # 检查坐标合法性
    if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height or x1 >= x2 or y1 >= y2:
        raise ValueError("cut_range 的坐标值非法或超出图像范围")

    # 裁剪图像
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image


if __name__ == '__main__':
    # 示例用法
    # image_paths = [
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_depth_show/IMG_2.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_depth_show/IMG_85.jpg',
    # ]  # 图像文件路径列表
    # stacked_image = stack_images(image_paths, direction='vertical', spacing=20)
    # stacked_image.save("depth.jpg")

    # image_paths = [
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_var_show/IMG_100.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_show/IMG_100.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_var_show/IMG_100.jpg',
    # ]
    # image_titles = ['使用KNN距离估计尺寸', '使用深度值估计尺寸', '结合KNN距离和深度值估计尺寸']
    # stacked_image = stack_images(image_paths, image_titles=image_titles, direction='horizontal', spacing=20,
    #                              bg_color="#ffffff")
    # stacked_image.save("head_size_comparison.jpg")

    # # 第一张对比图像
    # image_paths = [
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_var_show/IMG_100.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_show/IMG_100.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_var_show/IMG_100.jpg',
    # ]
    # image_titles = ['using KNN distance', 'using depth', 'combine']
    # stacked_image = stack_images(image_paths, image_titles=image_titles, direction='horizontal',
    #                              spacing=10,
    #                              bg_color="#ffffff",
    #                              font_size=16,
    #                              cut_range=[(450, 270), (650, 400)])
    # stacked_image.save("head_size_comparison_cut_en.jpg")
    #
    # # 第二张对比图像
    # image_paths = [
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_var_show/IMG_32.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_show/IMG_32.jpg',
    #     '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/pet_metric_depth/images_head_split_by_depth_var_show/IMG_32.jpg',
    # ]
    # stacked_image = stack_images(image_paths, image_titles=None, direction='horizontal',
    #                              spacing=10,
    #                              bg_color="#ffffff",
    #                              font_size=12,
    #                              cut_range=[(150, 300), (350, 430)])
    # stacked_image.save("head_size_comparison_cut_en2.jpg")
    #
    # # 合并两张对比图像
    # image_paths = [
    #     'head_size_comparison_cut_en.jpg',
    #     'head_size_comparison_cut_en2.jpg',
    # ]  # 图像文件路径列表
    # stacked_image = stack_images(image_paths, direction='vertical', spacing=10, bg_color="#ffffff")
    # stacked_image.save("head_size_comparison_cut_merge_en.jpg")

    # image_paths = [
    #     'depth_show_IMG_5.jpg',
    #     'depth_show_IMG_29.jpg',
    # ]  # 图像文件路径列表
    # stacked_image = stack_images(image_paths, direction='vertical', spacing=10)
    # stacked_image.save("depth_show_IMG_5_29_merge.jpg")

    image_paths = [
        '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/images/IMG_257.jpg',
        '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final/train_data/images/IMG_20.jpg',
    ]  # 图像文件路径列表
    stacked_image = stack_images(image_paths, direction='horizontal', spacing=10, bg_color="#ffffff")
    stacked_image.save("IMG_257_20_merge.jpg")

    print("merge sucess!")
