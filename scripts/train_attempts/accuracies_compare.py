import sys
sys.path.append(".")
import torch
from torch.nn import functional as F
from pathlib import Path
from src.dataset.datamodule import MGZDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet as UNetModel
from models.segformer import SegFormerModel
from models.vit_deep import Segmenter as ViTDeepModel
from models.deeplabV3 import DeepLabV3ResNet50 as DeepLabV3Model
from src.dataset.aug import *

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

            ROOT = Path.cwd()
            if input_channels == 1:
                processed_dir = ROOT / "data" / "processed"
                raw_dir = ROOT / "data" / "raw"
                datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=1, slice_size=(4, 4))
                datamodule.setup("fit")

                # Create DataLoader for validation set
                val_loader = DataLoader(datamodule.val_dataset, batch_size=1, shuffle=False)
            else:
                processed_dir = ROOT / "data" / "processedRGB"
                raw_dir = ROOT / "data" / "raw"
                transform_list = [
                    Blur(),
                    RandomHFlip(),
                    RandomVFlip(),
                    ToTensor()
                ]
                datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=8, slice_size=(4, 4), transform_list=transform_list)
                datamodule.prepare_data()
                datamodule.setup("fit")

                # Create DataLoader for validation set
                val_loader = DataLoader(datamodule.val_dataset, batch_size=8, shuffle=False)

            total_correct = 0
            total_pixels = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    if model_name == "ViTDeep" or model_name == "ViTDeep_unfrozen": 
                        resize_transform = transforms.Resize((224, 224))
                        images = resize_transform(images)
                        masks = resize_transform(masks)
                    images = images.to(device)
                    masks = masks.to(device).long()

                    outputs = model(images)
                    if model_name == "SegFormer":
                        outputs = F.interpolate(outputs, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
                    # elif model_name == "ViTDeep" or model_name == "ViTDeep_unfrozen":
                    #     outputs = F.interpolate(outputs, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
                    else:
                        outputs = F.softmax(outputs, dim=1)
                    predicted = outputs.argmax(1)

                    total_correct += (predicted == masks).sum().item()
                    total_pixels += masks.numel()

            accuracy = total_correct / total_pixels
            print(f"{model_name} Accuracy: {accuracy:.4f}")