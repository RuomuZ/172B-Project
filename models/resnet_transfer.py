from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn




class FCNResnetTransfer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=50, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        if "scale_factor" in kwargs:
            self.scale_factor = kwargs["scale_factor"]
        self.model = torch.hub.load(
            'pytorch/vision', 'fcn_resnet101', pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier[-1] = nn.Conv2d(512,
                                              self.out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=self.scale_factor)
        self.model = self.model.float()

    def forward(self, x):
        x = self.model(x.float())
        return x["out"]
