import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AugReg(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        super(AugReg, self).__init__()
        
        # Create the vision model from config
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        # Update the layers in the vision model
        self.model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(input_channels, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
        self.model.decode_head.classifier = torch.nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))
      
    def forward(self, x):
        x = self.model(x)
        return x.logits
