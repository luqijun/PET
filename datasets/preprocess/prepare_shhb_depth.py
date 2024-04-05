import os
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
from transformers import pipeline
from PIL import Image
import scipy.ndimage as ndimage
import cv2
from tqdm import tqdm
import h5py


def generate_depth_image(path):
     
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
     
    for folder in ['train_data', 'test_data']:
        images_path = os.path.join(path, folder, 'images')
        images_depth_path = os.path.join(path, folder, 'images_depth')
        os.makedirs(images_depth_path, exist_ok=True)
        for filename in tqdm(os.listdir(images_path)):
            image = Image.open(os.path.join(images_path, filename))
            depth_save_path = os.path.join(images_depth_path, filename)
            width, height = image.size
            
            if os.path.exists(depth_save_path):
                continue
           
            depth = pipe(image)["depth"]
            depth.save(depth_save_path)
            # depth_img = np.array(depth)


max_resolution = -1 # W*H ===> Train:(1024, 768) IMG_1.jpg Test:(1024, 768) IMG_1.jpg 

def get_max_resolution(path):
    global max_resolution
    
    
    for folder in ['test_data']:
        images_path = os.path.join(path, folder, 'images')
        # images_depth_path = os.path.join(path, folder, 'images_depth')
        # os.makedirs(images_depth_path, exist_ok=True)
        for filename in os.listdir(images_path):
            image = Image.open(os.path.join(images_path, filename))
            resolution = image.width * image.height
            if resolution > max_resolution:
                max_resolution = resolution
                print("max resolution = ", f'({str(image.width)}, {str(image.height)})', filename)


if __name__ == '__main__':
    
    # root_test = '/mnt/c/Users/lqjun/Desktop'
    
    root = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_B_final'
    # generate_depth_image(root)
    get_max_resolution(root)
    # print('Generate Success!')
    
    # get_max_min_value(root)
    pass
