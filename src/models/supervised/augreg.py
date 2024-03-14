import torch
from torch import nn
from torchvision.models import vit_b_32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AugReg(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        super(AugReg, self).__init__()
        
        # Create the vision model
        self.model = vit_b_32()

        # Update the layers in the vision model
        self.model.conv_proj = torch.nn.Conv2d(input_channels, 768, kernel_size=(32, 32), stride=(32, 32))
        self.model.heads.head = torch.nn.Linear(in_features=768, out_features=64, bias=True)
      
    def forward(self, x):
        x = self.model(x)
        return x
    