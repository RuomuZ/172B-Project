import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as torchvision_transforms
sys.path.append(".")

from src.file_utils.load_util import *
from src.dataset.subtile import Subtile

class MGZDataset(Dataset):
    def __init__(self, processed_dir: Path, 
                 transform: torchvision_transforms.transforms.Compose, 
                 slice_size: Tuple[int, int] = (4, 4)):
        self.subtile_dirs = []
        self.slice_size = slice_size
        self.transform = transform
        for tile in list((processed_dir / "subtiles").glob("*")):
            for subtile in list(tile.glob("*")):
                self.subtile_dirs.append(subtile)


    def __len__(self) -> int:
        return len(self.subtile_dirs)


    # def __getitem__(self, idx):
    #     subt = Subtile.load_subtile_by_dir(self.subtile_dirs[idx], self.slice_size)
    #     X = subt.image.values
    #     y = subt.mask.values
    #     result = self.transform({"X" : X, "y" : y})
    #     X = result["X"].permute(2, 0 ,1)
    #     y = result["y"].permute(2, 0 ,1)
    #     return (X, y)
    
    def __getitem__(self, idx):
        subt = Subtile.load_subtile_by_dir(self.subtile_dirs[idx], self.slice_size)
        X = subt.image.values
        y = subt.mask.values
        result = self.transform({"X" : X, "y" : y})
        X = result["X"].permute(2, 0 ,1)
        y = result["y"]
        labels_single_channel = np.full(y.shape[:2], 2, dtype=int)  # Default to class 2

        # Class 0: Red only (255, 0, 0)
        red_mask = (y[:, :, 0] == 255) & (y[:, :, 1] == 0) & (y[:, :, 2] == 0)
        labels_single_channel[red_mask] = 0

        # Class 1: Blue only (0, 0, 255)
        blue_mask = (y[:, :, 0] == 0) & (y[:, :, 1] == 0) & (y[:, :, 2] == 255)
        labels_single_channel[blue_mask] = 1
        y = labels_single_channel
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return (X, y)

    def displays(self):
        print(self.subtile_dirs)