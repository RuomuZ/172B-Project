import timm
import torch.nn as nn

class ViTModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, pretrained=True):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
        # Modify the input layer to handle different number of input channels if necessary
        if input_channels != 3:
            self.vit.patch_embed.proj = nn.Conv2d(input_channels, self.vit.patch_embed.embed_dim,
                                                  kernel_size=self.vit.patch_embed.patch_size, stride=self.vit.patch_embed.patch_size)

    def forward(self, x):
        return self.vit(x)

if __name__ == "__main__":
    model = ViTModel(input_channels=3, num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Should print torch.Size([1, 3])
