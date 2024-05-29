import sys
sys.path.append(".")
import numpy as np
from pathlib import Path
from halo import Halo
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset.datamodule import MGZDataModule
from models.unet import UNet
import logging
from sklearn.metrics import jaccard_score

logging.basicConfig(filename='trainingUNET.log', level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


sys.path.append(".")

ROOT = Path.cwd()

processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"

datamodule = MGZDataModule(
        processed_dir,
        raw_dir,
        batch_size=8,
        slice_size=(4, 4),
    )
print("got to prepare data")
datamodule.prepare_data()
datamodule.setup("fit")
train_dataset = datamodule.train_dataset
val_dataset = datamodule.val_dataset
# unique_labels = set()
# for _, labels in train_dataset:
#     unique_labels.update(np.unique(labels))
#     print(len(unique_labels))

# num_classes = len(unique_labels)
# print(num_classes)
# print(f"Number of classes: {num_classes}")

# Hyperparameters
learning_rate = 0.001
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
num_channels = 3 #RGB?
model = UNet(n_channels=num_channels, n_classes=3)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a DataLoader from the train_dataset
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    iou_score = 0
    val_loss = 0
    epoch_loss = 0
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        masks = masks.long()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks = masks.cpu().numpy()
        iou_score += jaccard_score(masks.flatten(), outputs.flatten(), average='macro')
    print("training done")
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    print("validation done")
    avg_loss = epoch_loss / len(train_loader)
    avg_iou_score = iou_score / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, IoU: {avg_iou_score}, Val Loss: {avg_val_loss}")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, IoU: {avg_iou_score}, Val Loss: {avg_val_loss}")

# Save the model checkpoint
torch.save(model.state_dict(), "unet_model.pth")