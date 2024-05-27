import sys
from pathlib import Path
import numpy as np
import cv2
import xarray as xr


sys.path.append(".")

#return a list of dirs of images and a list of dirs of masks
def process_file_name(file_path: Path):
    list_of_mask = list(file_path.glob('*_*'))
    list_of_data = [None] * len(list_of_mask)
    for i in range(len(list_of_mask)):
        prefix = list_of_mask[i].name.split("_")[0]
        list_of_data[i] = list(file_path.glob(f'{prefix}.*'))[0]
    return (list_of_data, list_of_mask)


#load a image into np.ndarray
def load_image(image_path):
    path_str = str(image_path)
    img = cv2.cvtColor(cv2.imread(path_str).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img


#load images and masks given dirs, and return two list of xr.array.
#list_of_data_dir and list_of_mask_dir have the same length
def load_images_masks(list_of_data_dir, list_of_mask_dir):
    X = []
    y = []
    for i in range(len(list_of_data_dir)):
        print(i)
        a_x = load_image(list_of_data_dir[i])
        a_y = load_image(list_of_mask_dir[i])
        if a_x.shape[0] < 3000:
            print(a_x.shape)
            a_x = np.swapaxes(a_x, 0, 1)
            a_y = np.swapaxes(a_y, 0, 1)
            print(f"after swap: {a_x.shape}")
        a_x = cv2.resize(a_x, (2480, 3508)).astype(np.float32)
        a_x = cv2.cvtColor(a_x, cv2.COLOR_BGR2RGB)
        a_y = cv2.resize(a_y, (2480, 3508)).astype(np.float32)
        a_y = cv2.cvtColor(a_y, cv2.COLOR_BGR2RGB)
        print(f"after resize: {a_x.shape}")
        x_array = xr.DataArray(
        data=a_x,
        dims=("height", "width", "band"),
        coords={
            "height": range(a_x.shape[0]),
            "width": range(a_x.shape[1]),
            "band": range(3)})
        x_array.attrs["id"] = i
        y_array = xr.DataArray(
        data=a_y,
        dims=("height", "width", "band"),
        coords={
            "height": range(a_y.shape[0]),
            "width": range(a_y.shape[1]),
            "band": range(3)})
        y_array.attrs["id"] = str(i)
        X.append(x_array)
        y.append(y_array)

    return X, y






