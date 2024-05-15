import sys

import numpy as np
from pathlib import Path
from halo import Halo
import matplotlib.pyplot as plt
sys.path.append(".")

from src.dataset.datamodule import MGZDataModule

ROOT = Path.cwd()

processed_dir = ROOT / "data" / "processed_exp3"
raw_dir = ROOT / "data" / "raw"

datamodule = MGZDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
        slice_size=(4, 4),
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
ax[0,0].imshow(train_feature)
ax[0,1].imshow(train_label)
plt.show()

