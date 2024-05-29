import numpy as np

def gt2mask(y):
    labels_single_channel = np.full(y.shape[:2], 2, dtype=int)  # Default to class 2

    # Class 0: Red only (255, 0, 0)
    red_mask = (y[:, :, 0] == 255) & (y[:, :, 1] == 0) & (y[:, :, 2] == 0)
    labels_single_channel[red_mask] = 0

    # Class 1: Blue only (0, 0, 255)
    blue_mask = (y[:, :, 0] == 0) & (y[:, :, 1] == 0) & (y[:, :, 2] == 255)
    labels_single_channel[blue_mask] = 1
    y = labels_single_channel
    return y