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

#You need .pth file to use this file
#import the models you are using above (I have imported unet and Deeplabv3)
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #=======
    #adjust the model accordingly
    #initialize the model with the parameters it trained on.
    #load the pth file to the model
    #model.eval()
    #This program is kind of hard code to use cuda, so there may be some bugs
    #if the model is not trained using cuda.
    model = UNet(n_channels=3, n_classes=3).to(device)
    state_dict = torch.load("unet_model.pth")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    #=======
    #adjust the path to the image accordingly
    #Path(ROOT / "data" / "test" / "1.jpg") is a example path
    #Also, If the model is trained on RGB dataset, please set the gray to be False, otherwise True
    #The size of the image does not need to be A4 size.
    #The slice size has to match the size you use to initialize the processed directory.
    img, processed_image, pred = load_and_process_test(
        model, Path(ROOT / "data" / "test" / "1.jpg"), 
        device, 
        (4, 4), gray=False)
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
    ax[0,0].imshow(img)
    ax[0,1].imshow(processed_image, cmap="gray")
    ax[0,2].imshow(pred)
    plt.show()

if __name__ == "__main__":
    main()