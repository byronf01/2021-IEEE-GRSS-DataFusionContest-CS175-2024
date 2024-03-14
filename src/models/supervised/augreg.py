import torch
from torch import nn
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AugReg(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        super(AugReg, self).__init__()
        
        # Create the vision model
        self.model = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, img_size=200)

        # Update the layers in the vision model
        self.model.patch_embed.proj = torch.nn.Conv2d(input_channels, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = torch.nn.Linear(in_features=768, out_features=64, bias=True)
      
    def forward(self, x):
        x = self.model(x)
        return x
