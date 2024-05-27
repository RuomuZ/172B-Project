import numpy as np

#mask in shape (C, W, H) -> ground truth in shape (W, H), class (0, 1, 2)
#0 -> white space, 1 -> blue (text), 2 -> red (picture)
def mask2gt(mask):
    if mask[0] == 0 and mask[1] == 0:
        return np.array([1.])
    elif mask[1] == 0 and mask[2] == 0:
        return np.array([2.])
    else:
        return np.array([0.])

