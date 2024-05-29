import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(".")
from src.dataset.subtile import Subtile
from src.dataset.datamodule import MGZDataModule
from src.dataset.aug import *
ROOT = Path.cwd()
processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"

transform_list = [
    Blur(),
    RandomHFlip(),
    RandomVFlip(),
]
datamodule = MGZDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
        slice_size=(4, 4),
    )

datamodule.prepare_data()
datamodule.setup("fit")
fig, ax = plt.subplots(1, 2, figsize=(4, 4), squeeze=False, tight_layout=True)
#restich the subtiles of image and mask together given the directory specified by the id
#The id is 80 here in the example
restich_dir = processed_dir / "Val" / "subtiles" / "80"
image, mask = Subtile.restich(restich_dir, (4, 4))
ax[0,0].imshow(np.uint8(image))
ax[0,1].imshow(mask)
print(mask)
plt.show()