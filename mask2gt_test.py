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
        slice_size=(2, 2),
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
train_label = train_label.reshape((train_feature.shape[0], train_feature.shape[1], 1))
print(train_feature.shape)
ax[0,0].imshow(train_feature)
ax[0,1].imshow(train_label)
plt.show()