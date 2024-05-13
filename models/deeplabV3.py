import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

class DeepLabV3ResNet50(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLabV3ResNet50, self).__init__()
        # Load pre-trained DeepLabV3 model with ResNet50 backbone
        self.model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        # Modify the classifier to change the number of output classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

if __name__ == "__main__":
    num_classes = 21  # Example for 21 classes
    model = DeepLabV3ResNet50(num_classes)
    
    # Example input tensor
    x = torch.randn(1, 3, 224, 224)  # Input size
    y = model(x)
    
    print(y.shape)  # Output tensor shape
