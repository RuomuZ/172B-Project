import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
sys.path.append(".")
from models.deeplabV3 import DeepLabV3ResNet50
from src.file_utils.load_test import *
from models.unet import UNet
ROOT = Path.cwd()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #adjust the model accordingly
    #model = DeepLabV3ResNet50(num_classes=3, num_channels=1)
    model = UNet(n_channels=3, n_classes=3).to(device)
    state_dict = torch.load("unet_model.pth")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    #adjust the path to the image accordingly
    img, processed_image, pred = load_and_process_test(
        model, Path(ROOT / "data" / "test" / "5.jpeg"), 
        device, 
        (4, 4), gray=False)
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
    ax[0,0].imshow(img)
    ax[0,1].imshow(processed_image)
    ax[0,2].imshow(pred)
    plt.show()

if __name__ == "__main__":
    main()