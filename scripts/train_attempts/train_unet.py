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

sys.path.append(".")

ROOT = Path.cwd()

processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"

datamodule = MGZDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
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
model = UNet(n_channels=num_channels, n_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a DataLoader from the train_dataset
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")

# Save the model checkpoint
torch.save(model.state_dict(), "unet_model.pth")