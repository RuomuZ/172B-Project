from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch

class SegFormerModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, model_name='nvidia/segformer-b0-finetuned-ade-512-512', **kwargs):
        super(SegFormerModel, self).__init__()
        self.input_conv = nn.Conv2d(input_channels, 3, kernel_size=1)  # New 1x1 conv layer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(model_name, **kwargs)
        
        # Adjust the classifier within the decode_head
        self.segformer.decode_head.classifier = nn.Conv2d(
            in_channels=self.segformer.decode_head.batch_norm.num_features,  # typically the number of input channels to the classifier
            out_channels=num_classes,  # your desired number of classes
            kernel_size=1,  # keeping the kernel size the same as the original
            stride=1  # keeping the stride the same as the original
        )

    def forward(self, x):
        x = self.input_conv(x) 
        outputs = self.segformer(x)
        return outputs.logits

# Example usage
if __name__ == "__main__":
    model = SegFormerModel(input_channels=3, num_classes=3)
    print(model)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)  # Should print torch.Size([1, 3, 512, 512])
