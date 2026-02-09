from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)       # Decoder features
        s1 = self.Ws(s)       # Skip connection features
        out = F.relu(g1 + s1) # Merge signals
        psi = self.psi(out)   # Attention map (0 to 1)
        return s * psi        # Filtered skip

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
    def __init__(self, input_channels=256, 
                 output_channels=256, 
                 bev_h=200, bev_w=200, 
                 channel_sizes=[256, 512, 1024, 2048],
                 loss_weights=dict(l1=1.0, msssim=0.0, std=0.0, gamma=1.0)):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        self.loss_weights = loss_weights

        # Encoder
        self.enc1 = EncoderBlock(input_channels, channel_sizes[0])
        self.enc2 = EncoderBlock(channel_sizes[0], channel_sizes[1])
        self.enc3 = EncoderBlock(channel_sizes[1], channel_sizes[2])

        # Bottleneck
        self.bottleneck = ConvBlock(channel_sizes[2], channel_sizes[3])

        # Decoder
        self.dec1 = DecoderBlock(channel_sizes[3], channel_sizes[2], channel_sizes[2])
        self.dec2 = DecoderBlock(channel_sizes[2], channel_sizes[1], channel_sizes[1])
        self.dec3 = DecoderBlock(channel_sizes[1], channel_sizes[0], channel_sizes[0])

        # Output layer
        self.final_conv = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)

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
        """
        Calculate loss.
        Args:
            preds (torch.Tensor): Predicted features, shape (B, N, C)
            targets (torch.Tensor): Target features, shape (B, N, C)
        Returns:
            dict: A dictionary of loss components.
        """
        losses = {}
        
        # 1. Reshape back to (B, C, H, W) for spatial losses
        bs, n_points, c = preds.shape
        p = preds.transpose(1, 2).view(bs, c, self.bev_h, self.bev_w)
        t = targets.transpose(1, 2).view(bs, c, self.bev_h, self.bev_w)

        # 2.STD weighted L1 Loss (on raw features)
        # weight_map = torch.std(t, dim=1, keepdim=True)
        # l1_val = torch.mean(self.l1_loss(p, t) * (1 + self.loss_weights['std'] * (torch.pow(weight_map, self.loss_weights['gamma']))))
        losses['bev_consumer_loss_l1loss'] = torch.mean(self.l1_loss(p, t)) * self.loss_weights['l1']

        # 3. MS-SSIM Loss (on Sigmoid-gated features)
        # We apply sigmoid to map unbounded features to [0, 1] for MS-SSIM stability
        # p_gated = torch.sigmoid(p)
        # t_gated = torch.sigmoid(t)

        # # ms_ssim expects (B, C, H, W). data_range=1.0 because of sigmoid.
        # # size_average=True returns a scalar loss.
        # # We use 1 - ms_ssim because we want to minimize the distance
        # msssim_val = 1 - ms_ssim(p_gated, t_gated, data_range=1.0, size_average=True)
        # losses['bev_consumer_loss_MS_SSIM'] = msssim_val * self.loss_weights['msssim']

        return losses

    def save_loss_gradient_map(self, preds, targets, epoch, batch_idx):
        """
        Visualizes where the loss is pulling the model.
        Bright spots = high influence from the Weighted STD map.
        """
        # 1. We need the gradient of the loss with respect to the prediction
        # We must ensure the prediction has grad enabled
        preds_for_grad = preds.detach().requires_grad_(True)
        
        # 2. Re-calculate loss for this specific pair
        loss_dict = self.loss(preds_for_grad, targets)
        loss = loss_dict['bev_consumer_loss_l1loss']
        
        # 3. Calculate gradients: dLoss / dPreds
        grad = torch.autograd.grad(loss, preds_for_grad)[0]
        
        # 4. Reshape to spatial (B, C, H, W) and take mean over channels
        bs, n, c = grad.shape
        grad_spatial = grad.transpose(1, 2).view(bs, c, 200, 200) # Using your bev_h/w
        grad_map = grad_spatial[0].mean(dim=0).cpu().numpy() # First sample in batch

        # 5. Plot with Robust Scaling (to avoid the "one color" issue)
        plt.figure(figsize=(6, 6))
        v_min, v_max = np.percentile(grad_map, [2, 98]) # Clip outliers
        
        plt.imshow(grad_map, cmap='magma', vmin=v_min, vmax=v_max)
        plt.colorbar(label="Gradient Intensity")
        plt.title(f"Loss Gradient Map (Spatial Influence)\nEpoch {epoch}")
        
        plt.savefig(f"spatial_influence_ep{epoch}_b{batch_idx}.png")
        plt.close()