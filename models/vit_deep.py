import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
import torch.onnx
from torchsummary import summary

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_1x1_output(x)
        return x

class Segmenter(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', num_classes=3, input_channels=3, freeze_backbone=False):
        super(Segmenter, self).__init__()
        self.backbone = create_model(backbone_name, pretrained=True)
        
        if input_channels != 3:
            self.backbone.patch_embed.proj = nn.Conv2d(input_channels, self.backbone.embed_dim, kernel_size=16, stride=16, dilation=2, padding=2)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.aspp = ASPP(self.backbone.embed_dim, self.backbone.embed_dim)
        self.linear_decoder = nn.Conv2d(self.backbone.embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone.forward_features(x)
        features = features[:, 1:]
        n_patches = features.shape[1]
        grid_size = int(n_patches ** 0.5)
        features = features.permute(0, 2, 1).view(B, -1, grid_size, grid_size)
        features = self.aspp(features)
        logits = self.linear_decoder(features)
        logits = nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits

# # Create a single dummy image and label for 3 classes with different input channels
# input_channels = 3  # Example for an image with 3 channels
# dummy_image = torch.randn(1, input_channels, 224, 224)  # Random image
# dummy_label = torch.randint(0, 3, (1, 224, 224))  # Random label for each pixel

# # Instantiate the model, loss function, and optimizer
# model = Segmenter(num_classes=3, input_channels=input_channels)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Training loop for a single image
# model.train()
# optimizer.zero_grad()
# output = model(dummy_image)
# print(f"Output shape: {output.shape}")
# output = output.permute(0, 2, 3, 1).reshape(-1, 3)  # Reshape for cross-entropy loss
# dummy_label = dummy_label.view(-1)  # Flatten labels
# loss = criterion(output, dummy_label)
# loss.backward()
# optimizer.step()

# print(f"Loss: {loss.item()}")

# # Testing the model on the same dummy image
# model.eval()
# with torch.no_grad():
#     test_output = model(dummy_image)
#     test_output = test_output.argmax(dim=1)
#     print(f"Predicted Label Shape: {test_output.shape}")  # Should match the input spatial dimensions (1, 224, 224)

# summary(model, (3, 224, 224))
