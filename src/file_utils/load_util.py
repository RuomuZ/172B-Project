import sys
from pathlib import Path
import numpy as np
import cv2
import xarray as xr


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
    return cv2.imread(path_str).astype(np.int16)


#load images and masks given dirs, and return two list of xr.array.
#list_of_data_dir and list_of_mask_dir have the same length
def load_images_masks(list_of_data_dir, list_of_mask_dir):
    X = []
    y = []
    for i in range(len(list_of_data_dir)):
        print(i)
        a_x = load_image(list_of_data_dir[i])
        a_y = load_image(list_of_mask_dir[i])
        x_array = xr.DataArray(
        data=a_x,
        dims=("height", "width", "band"),
        coords={
            "height": range(a_x.shape[0]),
            "width": range(a_x.shape[1]),
            "band": range(a_x.shape[2])})
        x_array.attrs["id"] = i
        y_array = xr.DataArray(
        data=a_y,
        dims=("height", "width", "band"),
        coords={
            "height": range(a_x.shape[0]),
            "width": range(a_x.shape[1]),
            "band": range(a_x.shape[2])})
        y_array.attrs["id"] = str(i)
        X.append(x_array)
        y.append(y_array)
    return X, y






