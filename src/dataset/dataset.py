import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as torchvision_transforms
import torchvision.transforms.functional as TF
sys.path.append(".")

from src.file_utils.load_util import *
from src.dataset.subtile import Subtile

class MGZDataset(Dataset):
    def __init__(self, processed_dir: Path, slice_size: Tuple[int, int] = (4, 4)):
        self.subtile_dirs = []
        self.slice_size = slice_size
        for tile in list((processed_dir / "subtiles").glob("*")):
            for subtile in list(tile.glob("*")):
                self.subtile_dirs.append(subtile)

#TO DO:
    def transform(self, image, mask):
        X = image
        y = mask
        return X, y


    def __len__(self) -> int:
        return len(self.subtile_dirs)


    def __getitem__(self, idx):
        subt = Subtile.load_subtile_by_dir(self.subtile_dirs[idx], self.slice_size)
        X = subt.image.values
        y = subt.mask.values
        X, y = self.transform(X, y)
        return (X, y)
    
    def displays(self):
        print(self.subtile_dirs)