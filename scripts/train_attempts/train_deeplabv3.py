import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import logging
from src.dataset.datamodule import MGZDataModule
from models.deeplabV3 import DeepLabV3ResNet50  # Importing the DeepLabV3 model from your file
from sklearn.metrics import jaccard_score

logging.basicConfig(filename='trainingDeepLabV3.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Setup data paths and datamodule
ROOT = Path.cwd()
processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"
datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=8, slice_size=(4, 4))  # Ensure slice_size matches DeepLabV3 input
datamodule.prepare_data()
datamodule.setup("fit")

# Create DataLoader
train_loader = DataLoader(datamodule.train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(datamodule.val_dataset, batch_size=8, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabV3ResNet50(num_classes=3) # Adjust num_classes as needed
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
total_iou = 0
# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    iou_score = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        masks = masks.long()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.cpu().numpy()
        iou_score += jaccard_score(masks.flatten(), outputs.flatten(), average='macro')
    
    # Calculate average IoU for the epoch
    iou_score /= len(train_loader)
    with torch.no_grad():
        val_loss = 0
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}, IoU: {iou_score}")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}, IoU: {iou_score}")
# Save the model
torch.save(model.state_dict(), "deeplabV3_model.pth")