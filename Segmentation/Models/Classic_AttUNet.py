import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
         
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        g1 = F.interpolate(g1, size = x1.shape[2:], mode = 'trilinear', align_corners=False)
        psi = self.relu(g1+x1)
        alpha = self.psi(psi)
        return x * alpha
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.conv(x)
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode = 'trilinear', align_corners=False)
        
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class AttUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.encoder1 = ConvBlock(in_channels, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.encoder4 = ConvBlock(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Attention Gates
        self.att3 = AttentionBlock(gate_channels=256, skip_channels=128, inter_channels=64)
        self.att2 = AttentionBlock(gate_channels=128, skip_channels=64, inter_channels=32)
        self.att1 = AttentionBlock(gate_channels=64, skip_channels=32, inter_channels=16)

        # Decoder
        self.up3 = UpBlock(256 + 128, 128)
        self.up2 = UpBlock(128 + 64, 64)
        self.up1 = UpBlock(64 + 32, 32)

        # Final output layer
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        #Encoders
        enc1 = self.encoder1(x)
        x = F.max_pool3d(enc1, kernel_size=2)
        
        enc2 = self.encoder2(x)
        x = F.max_pool3d(enc2, kernel_size=2)
        
        enc3 = self.encoder3(x)
        x = F.max_pool3d(enc3, kernel_size=2)
        
        enc4 = self.encoder4(x)
        x = F.max_pool3d(enc4, kernel_size=2)
        
        #Bottleneck
        x = self.bottleneck(x)
        
        #Decoders
        att3 = self.att3(g=x, x=enc4)
        x = torch.cat([x, att3], dim=1)
        x = self.up3(x)
        
        att2 = self.att2(g=x, x=enc3)
        x = torch.cat([x, att2], dim=1)
        x = self.up2(x)
        
        att1 = self.att1(g=x, x=enc2)
        x = torch.cat([x, att1], dim=1)
        x = self.up1(x)
        
        #Final output layer
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners = False)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x