import torch
from torch import nn
from transformers import ViTConfig, ViTModel
import models.AugReg.upernet_augreg_adapter_tiny_512_160k_ade20k

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AugReg(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        super(AugReg, self).__init__()
        
        # Create the vision model from config
        self.config = ViTConfig.from_dict(models.AugReg.upernet_augreg_adapter_tiny_512_160k_ade20k.model)
        self.config.auxiliary_head['in_channels'] = input_channels
        self.config.num_channels = input_channels
        self.config.image_size = 200

        self.model = ViTModel(self.config)

        # Update the layers in the vision model
        self.model.pooler.dense = torch.nn.Linear(in_features=768, out_features=16*output_channels, bias=True)
      
    def forward(self, x):
        x = self.model(x)
        return x.pooler_output
