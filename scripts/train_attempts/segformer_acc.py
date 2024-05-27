import sys
sys.path.append(".")
import torch
from src.dataset.datamodule import MGZDataModule
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from models.segformer import SegFormerModel  # Assuming your model is in this path
if __name__ == '__main__':
    ROOT = Path.cwd()
    processed_dir = ROOT / "data" / "processed"
    raw_dir = ROOT / "data" / "raw"
    datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=1, slice_size=(4, 4))
    datamodule.setup("fit")

    # Create DataLoader for validation set
    val_loader = DataLoader(datamodule.val_dataset, batch_size=1, shuffle=False)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegFormerModel(input_channels=3, num_classes=3).to(device)
    model.load_state_dict(torch.load("segformer_model.pth"))  # Load trained model
    model.eval()

    # Validation loop
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()  # Ensure masks are long type for comparison

            # Forward pass
            outputs = model(images)
            outputs = F.interpolate(outputs, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
            predicted = outputs.argmax(1)  # Get the class with the highest probability

            total_correct += (predicted == masks).sum().item()
            total_pixels += masks.numel()

    # Calculate accuracy
    accuracy = total_correct / total_pixels
    print(f"Validation Accuracy: {accuracy:.4f}")
