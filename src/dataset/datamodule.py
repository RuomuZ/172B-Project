import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import xarray as xr
from sklearn.model_selection import train_test_split
from torchvision import transforms as torchvision_transforms
from tqdm import tqdm


sys.path.append(".")

from src.file_utils.load_util import *
from src.dataset.subtile import Subtile
from src.dataset.dataset import MGZDataset
from src.dataset.aug import *


def collate_fn(batch):
    Xs = []
    ys = []
    for X, y in batch:
        Xs.append(X)
        ys.append(y)

    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return Xs, ys


class MGZDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processed_dir: Path,
        raw_dir: Path,
        batch_size: int = 32,
        seed: int = 12378921,
        slice_size: Tuple[int, int] = (4, 4),
        train_size: float = 0.8,
        transform_list: list = [
            torchvision_transforms.RandomApply([AddNoise()], p=0.5),
            torchvision_transforms.RandomApply([Blur()], p=0.5),
            torchvision_transforms.RandomApply([RandomHFlip()], p=0.5),
            torchvision_transforms.RandomApply([RandomVFlip()], p=0.5),
            ToTensor(),
        ],
        num_workers = 3
    ):
        super(MGZDataModule, self).__init__()
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.batch_size = batch_size
        self.seed = seed
        self.slice_size = slice_size
        self.train_size = train_size
        self.train_dir = self.processed_dir / "Train"
        self.val_dir = self.processed_dir / "Val"
        self.transform = torchvision_transforms.transforms.Compose(transform_list)
        self.num_workers = num_workers


    def load_and_preprocess(self, dir: Path):
        X, y = process_file_name(dir)
        X_array, y_array = load_images_masks(X, y)
        return X_array, y_array
    
    
    def prepare_data(self) -> None:
        if not self.processed_dir.exists() or len([*(self.processed_dir).iterdir()]) == 0:
            X, y = self.load_and_preprocess(self.raw_dir)
            train_n = int(len(X) * self.train_size)
            X_train = X[:train_n]
            y_train = y[:train_n]
            X_test = X[train_n:]
            y_test = y[train_n:]
            for idx in tqdm(range(len(X_train)), desc="Processing train tiles"):
                subt = Subtile(X_train[idx], y_train[idx], self.slice_size)
                subt.save(self.train_dir)
            for idx in tqdm(range(len(X_test)), desc="Processing validation tiles"):
                subt = Subtile(X_test[idx], y_test[idx], self.slice_size)
                subt.save(self.val_dir)


    def setup(self, stage: str) -> None:
        if stage == "fit":
            print(self.train_dir, self.val_dir)
            self.train_dataset = MGZDataset(self.train_dir, self.transform, self.slice_size)
            self.val_dataset = MGZDataset(self.val_dir, self.transform, self.slice_size)


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers = self.num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers = self.num_workers)