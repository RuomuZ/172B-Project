import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from src.dataset.datamodule import MGZDataModule
from models.vit_deep import Segmenter  # Importing the ViT model from your file
from src.dataset.aug import *

# Setup data paths and datamodule
ROOT = Path.cwd()
processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"
datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=1, slice_size=(4, 4))  # Ensure slice_size matches ViT input
datamodule.prepare_data()
datamodule.setup("fit")
resize_transform = transforms.Resize((224, 224))

# Create DataLoader
train_loader = DataLoader(datamodule.train_dataset, batch_size=1, shuffle=True)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Segmenter(input_channels=3, num_classes=3).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images = resize_transform(images)
        images = images.to(device)
        masks = resize_transform(masks)
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}")

# Save the model
torch.save(model.state_dict(), "vitdeep_model.pth")