import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, img_size=(96, 96, 96)):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=False
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))