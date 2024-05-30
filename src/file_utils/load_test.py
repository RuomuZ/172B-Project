import sys
from pathlib import Path
import numpy as np
import cv2
import xarray as xr
import torch

sys.path.append(".")

def calculate_slice_index(x: int, y: int, slice_size: tuple, length: tuple, ):
    # calculate the start and end indices for the slice based
    # on the slice_size and the x and y coordinates
    start_index = (
        int(np.divide(length[0], slice_size[0]) * x),
        int(np.divide(length[1], slice_size[1]) * y),
    )
    end_index = (
        int(np.divide(length[0], slice_size[0]) * (x + 1)),
        int(np.divide(length[1], slice_size[1]) * (y + 1)),
    )

    if start_index[0] > length[0] or start_index[1] > length[1]:
        raise IndexError(
            f"Start index {start_index} out of range for img of shape {length}"
        )

    if end_index[0] > length[0] or end_index[1] > length[1]:
        raise IndexError(
            f"End index {end_index} out of range for img of shape {length}"
        )

    return start_index, end_index

def get_subtile_from_parent_image(image, x: int, y: int, slize_size) -> xr.Dataset:
    img_length = (image.shape[0], image.shape[1])
    start_index_img, end_index_img = calculate_slice_index(
        x, y, slize_size, img_length
    )
    sliced_data_array = image[
            start_index_img[0] : end_index_img[0],
            start_index_img[1] : end_index_img[1],
            :
        ]
    return sliced_data_array



#load a image into np.ndarray
def load_image(image_path):
    path_str = str(image_path)
    img = cv2.cvtColor(cv2.imread(path_str).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img


def load_and_process_test(model, data_dir, device, slice_size = (4, 4), gray=False, resize_to=None):
    original_image = load_image(data_dir)
    a_x = original_image.copy()
    processed_image = cv2.cvtColor(a_x, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    if gray:
        a_x = np.expand_dims(processed_image, axis=2)
    row_x = []
    for x in range(slice_size[0]):
        col_x = []
        for y in range(slice_size[1]):
            subtile = (get_subtile_from_parent_image(a_x, x, y, slice_size))
            original_subtile_size = subtile.shape[:2]
            if resize_to is not None:
                subtile = cv2.resize(subtile, resize_to).astype(np.float32)
            if gray:
                subtile = torch.from_numpy(subtile).squeeze(dim=2).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            else:
                subtile = torch.from_numpy(subtile).permute(2, 0, 1).unsqueeze(dim=0).to(device)
            subtile = subtile.float()  # Convert to float  # Convert to float
            pred = model(subtile)
            pred = pred.detach().cpu()
            pred = np.uint8(torch.squeeze(pred, 0).argmax(dim=0))
            if resize_to is not None:
                pred = cv2.resize(pred, original_subtile_size[::-1])  # Resize back to original size
            col_x.append(pred)
        row_x.append(np.concatenate(col_x, axis=1))
    full_mask = np.concatenate(row_x, axis=0)
    return original_image, processed_image ,full_mask