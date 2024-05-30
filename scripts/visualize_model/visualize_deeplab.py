import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import pytorch_lightning as pl
import wandb
import torch
import numpy as np
from pytorch_lightning.loggers import WandbLogger
sys.path.append(".")
from src.dataset.datamodule import MGZDataModule
from models.deeplabV3 import DeepLabV3ResNet50
from src.utilities import ESDConfig, PROJ_NAME
ROOT = Path.cwd()

processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"
from src.dataset.aug import Blur, RandomHFlip, RandomVFlip, ToTensor


def main():
    datamodule = MGZDataModule(processed_dir,
        raw_dir,
        batch_size=1,
        slice_size=(4, 4)
    )
    datamodule.prepare_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3ResNet50(num_classes=3, num_channels=1)
    model.load_state_dict(torch.load("deeplabV3_model1.pth"))
    model.eval()
    model = model.to(device)
    datamodule.setup("fit")
    eval_dataset = datamodule.val_dataset
    img, mask = eval_dataset[30]

    img, mask = img.unsqueeze(0).to(device), mask.to(device)
    pred = model(img)
    print(mask)
    print(pred)
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
    img = img.detach().cpu()
    mask = mask.detach().cpu()
    pred = pred.detach().cpu()
    print(type(img))
    img = np.uint8(torch.squeeze(img, 0).permute(1, 2, 0))
    pred = np.uint8(torch.squeeze(pred, 0).argmax(dim=0))
    ax[0,0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0,1].imshow(mask)
    ax[0,2].imshow(pred)
    plt.show()

if __name__ == "__main__":
    main()