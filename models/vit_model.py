import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model

class Segmenter(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', num_classes=3, input_channels=3):
        super(Segmenter, self).__init__()
        self.backbone = create_model(backbone_name, pretrained=True)
        
        # Adjust the first convolution layer to accept different input channels
        if input_channels != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_channels, self.backbone.embed_dim, kernel_size=16, stride=16)

        self.num_classes = num_classes
        self.linear_decoder = nn.Conv2d(self.backbone.embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        B, C, H, W = x.shape
        features = self.backbone.forward_features(x)  # Extract features using ViT
        # print(f"Features shape: {features.shape}")
        # Exclude the classification token
        features = features[:, 1:]
        # print(f"Features shape without classification token: {features.shape}")
        n_patches = features.shape[1]
        # Calculate the spatial dimensions of the patch grid
        grid_size = int(n_patches ** 0.5)
        features = features.permute(0, 2, 1).view(B, -1, grid_size, grid_size)  # Reshape to 2D feature map
        # print(f"Reshaped features shape: {features.shape}")
        logits = self.linear_decoder(features)  # Linear decoding to get class scores for each pixel
        # print(f"Logits shape before upsampling: {logits.shape}")
        logits = nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)  # Upsample to match input size
        # print(f"Logits shape after upsampling: {logits.shape}")
        return logits

# Create a single dummy image and label for 3 classes with different input channels
input_channels = 3  # Example for an image with 3 channels
dummy_image = torch.randn(1, input_channels, 224, 224)  # Random image
dummy_label = torch.randint(0, 3, (1, 224, 224))  # Random label for each pixel

# Instantiate the model, loss function, and optimizer
model = Segmenter(num_classes=3, input_channels=input_channels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop for a single image
model.train()
optimizer.zero_grad()
output = model(dummy_image)
print(f"Output shape: {output.shape}")
output = output.permute(0, 2, 3, 1).reshape(-1, 3)  # Reshape for cross-entropy loss
dummy_label = dummy_label.view(-1)  # Flatten labels
loss = criterion(output, dummy_label)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")

# Testing the model on the same dummy image
model.eval()
with torch.no_grad():
    test_output = model(dummy_image)
    test_output = test_output.argmax(dim=1)
    print(f"Predicted Label Shape: {test_output.shape}")  # Should match the input spatial dimensions (1, 224, 224)
