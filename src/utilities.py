from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List
import xarray as xr

ROOT = Path.cwd()
PROJ_NAME = "172B_Project"
MODEL =  "FCNResnetTransfer"

@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed"
    raw_dir: Path = ROOT / "data" / "raw"
    results_dir: Path = ROOT / "data" / "predictions" / MODEL
    accelerator: str = "gpu"
    batch_size: int = 2
    depth: int = 2
    devices: int = 1
    embedding_size: int = 64
    in_channels: int = 3
    kernel_size: int = 3
    learning_rate: float = 1e-3
    max_epochs: int = 10
    model_path: Path = ROOT / "models" / MODEL / "last.ckpt"
    model_type: str = MODEL
    n_encoders: int = 2
    num_workers: int = 7
    out_channels: int = 3 
    pool_sizes: str = "5,5,2"
    seed: int = 12378921
    slice_size: tuple = (4, 4)


