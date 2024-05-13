from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

class SegFormerModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, model_name='nvidia/segformer-b0-finetuned-ade-512-512', **kwargs):
        super(SegFormerModel, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_classes, **kwargs)
        
        # Modify the input layer if necessary
        if input_channels != 3:
            self.segformer.model.config.num_channels = input_channels
            self.segformer.model.encoder.embeddings.patch_embeddings.projection = nn.Conv2d(
                input_channels, 
                self.segformer.model.config.hidden_sizes[0], 
                kernel_size=self.segformer.model.config.patch_size, 
                stride=self.segformer.model.config.patch_size, 
                padding=0, 
                bias=False
            )

    def forward(self, x):
        outputs = self.segformer(x)
        return outputs.logits

if __name__ == "__main__":
    model = SegFormerModel(input_channels=3, num_classes=3)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)  # Should print torch.Size([1, 3, 512, 512])

