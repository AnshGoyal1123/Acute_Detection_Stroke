import torch
import torch.nn as nn
from monai.networks.nets import UNet

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=0  # standard UNet, no residuals
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))