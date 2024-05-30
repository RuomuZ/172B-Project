
import sys
sys.path.append(".")
import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from src.file_utils.load_test import *
from models.unet import UNet as UNetModel
from models.segformer import SegFormerModel
from models.vit_deep import Segmenter as ViTDeepModel
from models.deeplabV3 import DeepLabV3ResNet50 as DeepLabV3Model
from src.dataset.aug import *

# Assume image_files is a list of image file paths
image_files = ["1.jpg", "2.png", "3.jpg", "4.jpg", "5.jpeg", "putin_img.jpg"]
# image_files = ["putin_img.jpg"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
    "SegFormer": SegFormerModel,
    "ViTDeep": ViTDeepModel,
    "ViTDeep_unfrozen": ViTDeepModel,
    "UNet": UNetModel,
    "DeepLabV3": DeepLabV3Model
}

pth_files = {
    "1ch": {
        "UNet": "unet_model_1ch.pth",
        "DeepLabV3": "deeplabV3_model_1ch.pth"
    },
    "3ch": {
        "SegFormer": "segformer_model_3ch.pth",
        "ViTDeep": "vitdeep_model_3ch.pth",
        "ViTDeep_unfrozen": "vitdeep_model_3ch_unfrozen.pth",
        "UNet": "unet_model_3ch.pth",
        "DeepLabV3": "deeplabV3_model_3ch.pth"
    }
}

for ch, model_files in pth_files.items():
    print(f"Evaluating {ch} models")
    ROOT = Path.cwd()
    input_channels = 1 if ch == "1ch" else 3
    for model_name, model_class in models.items():
        if model_name in model_files:
            print(model_name)
            if model_name == "DeepLabV3":
                model = model_class(num_channels=input_channels, num_classes=3).to(device)
            elif model_name == "UNet":
                model = model_class(n_channels=input_channels, n_classes=3).to(device)
            elif model_name == "ViTDeep":
                model = model_class(input_channels=input_channels, num_classes=3, freeze_backbone=True).to(device)
            elif model_name == "ViTDeep":
                model = model_class(input_channels=input_channels, num_classes=3, freeze_backbone=False).to(device)
            else:
                model = model_class(input_channels=input_channels, num_classes=3).to(device)
            state_dict = torch.load(model_files[model_name])
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()
            if input_channels == 1:
                gray = True
            else:
                gray = False
            resize = None
            for image_file in image_files:
                if model_name == "ViTDeep" or model_name == "ViTDeep_unfrozen": 
                    resize = (224, 224)
                img, processed_image, pred = load_and_process_test(
                    model, Path(ROOT / "data" / "test" / image_file), 
                    device, 
                    (4, 4), 
                    gray=gray,
                    resize_to=resize)
                fig, ax = plt.subplots(1, 3, figsize=(4, 4), squeeze=False, tight_layout=True)
                ax[0,0].imshow(img)
                ax[0,1].imshow(processed_image, cmap="gray")
                ax[0,2].imshow(pred)

                image_file_name, _ = os.path.splitext(image_file)

                # Save the plot in the 'preds' folder
                plt.savefig(f'preds/{model_name}_{ch}_{image_file_name}_prediction.png')

                plt.show()
                plt.close(fig)

                            # Load image
                # image = cv2.imread(image_file)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = transforms.ToTensor()(image).unsqueeze(0).to(device)

                # Resize if necessar

                # # Make prediction
                # with torch.no_grad():
                #     output = model(image)
                #     if model_name == "SegFormer":
                #         output = F.interpolate(output, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                #     else:
                #         output = F.softmax(output, dim=1)
                #     predicted = output.argmax(1)

                # # Convert prediction to numpy array
                # predicted_np = predicted.cpu().squeeze().numpy()

                # Plot original image and prediction
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB))
                # ax[0].set_title('Original Image')
                # ax[1].imshow(predicted_np)
                # ax[1].set_title(f'{model_name} Prediction')
                # plt.savefig(f'preds/{model_name}_{ch}_prediction.png')
                # plt.show()
                

                