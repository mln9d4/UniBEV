from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)       # Decoder features
        s1 = self.Ws(s)       # Skip connection features
        out = F.silu(g1 + s1) # Merge signals
        psi = self.psi(out)   # Attention map (0 to 1)
        return s * psi        # Filtered skip

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
            skip = self.conv(x)   # Before pooling - for skip connection
            down = self.pool(skip) # After pooling - goes deeper
            return skip, down

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.att = AttentionGate(in_channels, skip_channels, out_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
  
    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        skip = self.att(x, skip)        # Filter skip with attention
        x = torch.cat([x, skip], dim=1) # Merge
        return self.conv(x)

@HEADS.register_module()
class UNetAttention(BaseModule):
    def __init__(self, input_channels=256, output_channels=256, bev_h=200, bev_w=200):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Encoder
        self.enc1 = EncoderBlock(input_channels, 256)
        self.enc2 = EncoderBlock(256, 512)
        self.enc3 = EncoderBlock(512, 1024)

        # Bottleneck
        self.bottleneck = ConvBlock(1024, 2048)

        # Decoder
        self.dec1 = DecoderBlock(2048, 1024, 1024)
        self.dec2 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)

        # Output layer
        self.final_conv = nn.Conv2d(256, output_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: (bs, bev_h * bev_w, C)
        # Reshape to (bs, C, bev_h, bev_w) for Conv2d
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        b1 = self.bottleneck(p3)
        d1 = self.dec1(b1, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)

        x = self.final_conv(d3)

        # Back to (bs, bev_h * bev_w, output_channels)
        x = x.flatten(2).transpose(1, 2)
        return x

    def loss(self, preds, targets):
        losses = {}

        loss = self.loss_fn(preds, targets)
        losses['bev_consumer_l1_loss'] = loss

        return losses