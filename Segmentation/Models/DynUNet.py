import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

class DynUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3, 3]] * 6,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 5,
            norm_name="instance",
            deep_supervision=False,
            res_block=False,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))