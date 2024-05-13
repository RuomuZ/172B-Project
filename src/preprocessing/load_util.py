import sys
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
import xarray as xr
import tifffile
import cv2


test_path = Path("/Users/zhaoruomu/Documents/GitHub/172B-Project/data/raw")
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
def load_images(image_path):
    path_str = str(image_path)
    print(path_str)
    file_extension = path_str.split("/")[-1].split(".")[-1]
    if file_extension == "tif":
        return tifffile.imread(path_str, dtype=np.float32)
    else:
        return cv2.imread(path_str)



#for test purpose
#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 2, figsize=(4, 4), squeeze=False, tight_layout=True)
    # --- start here ---
    # imshow the ground truth image
#x, y = process_file_name(test_path)
#data_array = load_images(x[0])
#data_array2 = load_images(y[0])
#print(data_array.shape)
#print(data_array2.shape)
#ax[0,0].imshow(data_array)
#ax[0,1].imshow(data_array2)
#plt.show()




