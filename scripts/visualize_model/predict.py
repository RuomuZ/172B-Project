import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
sys.path.append(".")
from models.deeplabV3 import DeepLabV3ResNet50
from src.file_utils.load_test import *
ROOT = Path.cwd()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3ResNet50(num_classes=3, num_channels=1)
    model.load_state_dict(torch.load("deeplabV3_model1.pth"))
    model.eval()
    model = model.to(device)
    img, processed_image, pred = load_and_process_test(model, Path(ROOT / "data" / "test" / "1.jpg"), device, (4, 4))
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
    ax[0,0].imshow(img)
    ax[0,1].imshow(processed_image, cmap='gray', vmin=0, vmax=255)
    ax[0,2].imshow(pred)
    plt.show()

if __name__ == "__main__":
    main()