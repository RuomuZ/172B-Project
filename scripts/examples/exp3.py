import sys

import numpy as np
from pathlib import Path
from halo import Halo
import matplotlib.pyplot as plt
from torchvision import transforms
sys.path.append(".")

from src.dataset.datamodule import MGZDataModule
from src.dataset.aug import Blur, RandomHFlip, RandomVFlip, ToTensor
ROOT = Path.cwd()

processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"


transform_list = [
    Blur(),
    RandomHFlip(),
    RandomVFlip(),
    ToTensor()
]
datamodule = MGZDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
        slice_size=(4, 4),
        transform_list=transform_list
    )

datamodule.prepare_data()
datamodule.setup("fit")
tr = datamodule.train_dataset
te = datamodule.val_dataset
print(len(tr))
print(len(te))
# Display image and label.
train_feature, train_label = tr[2]
print(f"Feature batch shape: {train_feature.shape}")
print(f"Labels batch shape: {train_label.shape}")
#label = train_labels[0].squeeze()
fig, ax = plt.subplots(1, 2, figsize=(4, 4), squeeze=False, tight_layout=True)
train_feature = np.uint8(train_feature.permute(1, 2, 0))
train_label = np.uint8(train_label.permute(1, 2, 0))
print(train_label.shape)
print(train_feature.shape)
ax[0,0].imshow(train_feature)
ax[0,1].imshow(train_label)
plt.show()
# Assuming `labels` is your (877, 620, 3) numpy array
# Assuming `labels` is your RGB array of shape (877, 620, 3)
labels_single_channel = np.full(train_label.shape[:2], 2, dtype=int)  # Default to class 2

# Class 0: Red only (255, 0, 0)
red_mask = (train_label[:, :, 0] == 255) & (train_label[:, :, 1] == 0) & (train_label[:, :, 2] == 0)
labels_single_channel[red_mask] = 0

# Class 1: Blue only (0, 0, 255)
blue_mask = (train_label[:, :, 0] == 0) & (train_label[:, :, 1] == 0) & (train_label[:, :, 2] == 255)
labels_single_channel[blue_mask] = 1

print(labels_single_channel.shape)

# Display the result
plt.imshow(labels_single_channel, cmap='tab20')
plt.colorbar()
plt.show()