import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTSegmentation(nn.Module):
    """Segmentation model using Vision Transformer as encoder."""
    def __init__(self, num_channels, num_classes):
        super(ViTSegmentation, self).__init__()
        self.num_classes = num_classes
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Modify the head for the number of segmentation classes
        num_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Linear(num_features, num_classes)

        # Adjust the decoder to correctly upsample to the original image size
        # Adding more layers to upscale from 16x16 to 256x256
        self.up = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)   # 128 -> 256
        )

    def forward(self, x):
        # Adjust input size for ViT
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vit(x)  # Output shape: (batch_size, num_classes)
        print(x.shape)

        # Reshape to add height and width dimensions
        x = x.view(x.shape[0], self.num_classes, 1, 1)
        print(x.shape)

        # Upsample to original size
        x = self.up(x)
        print(x.shape)
        return x

# Example usage:
model = ViTSegmentation(num_channels=3, num_classes=10)
input_tensor = torch.randn(1, 3, 256, 256)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 10, 256, 256])

# import torch
# import torch.nn as nn
# from torchvision.models import resnet50
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SimpleSegmenter(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleSegmenter, self).__init__()
#         self.backbone = resnet50(pretrained=True)
#         self.transformer_encoder_layer = TransformerEncoderLayer(d_model=2048, nhead=8)
#         self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
#         self.decoder = nn.Conv2d(2048, num_classes, kernel_size=1)

#     def forward(self, x):
#         features = self.backbone(x)
#         features = self.transformer_encoder(features.flatten(2).permute(2, 0, 1))
#         features = features.permute(1, 2, 0).view(x.shape[0], 2048, x.shape[-2], x.shape[-1])
#         out = self.decoder(features)
#         return out

# # Example usage:
# model = SimpleSegmenter(num_classes=21)
# input_tensor = torch.rand(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)

