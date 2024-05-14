import sys
from pathlib import Path
import numpy as np
import tifffile
import cv2

sys.path.append(".")

#return a list of dirs of images and a list of dirs of masks
def process_file_name(file_path: Path):
    list_of_mask = list(file_path.glob(f'*_*'))
    list_of_data = [None] * len(list_of_mask)
    for i in range(len(list_of_mask)):
        prefix = str(list_of_mask[i]).split("/")[-1].split("_")[0]
        list_of_data[i] = list(file_path.glob(f'{prefix}.*'))[0]
    return (list_of_data, list_of_mask)

#load a image into np.ndarray
def load_image(image_path):
    path_str = str(image_path)
    file_extension = path_str.split("/")[-1].split(".")[-1]
    if file_extension == "tif":
        print("here")
        return tifffile.imread(path_str, chunkdtype=np.float32)
    else:
        return cv2.imread(path_str)

#load images and masks given dirs, and return a tuple of two list.
#list_of_data_dir and list_of_mask_dir have the same length
def load_images_masks(list_of_data_dir, list_of_mask_dir):
    X = []
    y = []
    for i in range(len(list_of_data_dir)):
        X.append(load_image(list_of_data_dir[i]))
        y.append(load_image(list_of_mask_dir[i]))
    return X,y






