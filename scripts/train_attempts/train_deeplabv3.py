import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from src.dataset.datamodule import MGZDataModule
from models.deeplabV3 import DeepLabV3ResNet50  # Importing the DeepLabV3 model from your file

# Setup data paths and datamodule
ROOT = Path.cwd()
processed_dir = ROOT / "data" / "processed"
raw_dir = ROOT / "data" / "raw"
datamodule = MGZDataModule(processed_dir, raw_dir, batch_size=1, slice_size=(224, 224))  # Ensure slice_size matches DeepLabV3 input
datamodule.prepare_data()
datamodule.setup("fit")

# Create DataLoader
train_loader = DataLoader(datamodule.train_dataset, batch_size=1, shuffle=True)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabV3ResNet50(num_classes=3).to(device)  # Adjust num_classes as needed

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

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
torch.save(model.state_dict(), "deeplabV3_model.pth")