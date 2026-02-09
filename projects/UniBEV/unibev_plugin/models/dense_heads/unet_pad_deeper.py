from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np

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


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        return x

@HEADS.register_module()
class UnetPadDeeper(BaseModule):
    """
    UNet with input Padding and center cropping for the output with 5 layers, mirroring what the original UNet paper did.
    2**n rule, n=4 here, so input size should be multiple of 16. Pad input to 208x208.
    So we get this: 208 -> 104 -> 52 -> 26 -> 13 -> 26 -> 52 -> 104 -> 208
    Centercrop output to 200x200.
    
    """
    def __init__(self, input_channels=256, output_channels=256, 
                channel_sizes=[256, 256, 256, 256],
                bev_h=200, bev_w=200, 
                loss_weights=dict(l1=0.16, msssim=0.84, std=20.0, l1_regularization=1e-2)):
        super().__init__()
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        # Zhao paper says 0.16 for l1 and 0.84 for ms-ssim
        self.loss_weights = loss_weights  # Only L1, MS-SSIM, std, and l1_regularization
        self.l1_loss = nn.L1Loss(reduction='none')


        # Encoder blocks (contracting path)
        self.enc1 = ConvBlock(input_channels, channel_sizes[0])
        self.enc2 = ConvBlock(channel_sizes[0], channel_sizes[1])
        self.enc3 = ConvBlock(channel_sizes[1], channel_sizes[2])
        self.enc4 = ConvBlock(channel_sizes[2], channel_sizes[3])    

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(channel_sizes[3], channel_sizes[3]*2)

        self.deconv1 = DeconvBlock(channel_sizes[3]*2, channel_sizes[3])
        self.deconv2 = DeconvBlock(channel_sizes[3], channel_sizes[2])
        self.deconv3 = DeconvBlock(channel_sizes[2], channel_sizes[1])
        self.deconv4 = DeconvBlock(channel_sizes[1], channel_sizes[0])

        self.out = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)

    def forward(self, x):
        # Pad input to be multiple of 16
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        x = F.pad(x, (4, 4, 4, 4), mode='constant', value=0)  # Pad to 208x208

        # Encoder
        enc_out1 = self.enc1(x)
        down1 = self.pool1(enc_out1)

        enc_out2 = self.enc2(down1)
        down2 = self.pool2(enc_out2)

        enc_out3 = self.enc3(down2)
        down3 = self.pool3(enc_out3)

        enc_out4 = self.enc4(down3)
        down4 = self.pool4(enc_out4)
        # Bottleneck
        bottleneck = self.bottleneck(down4)

        # Decoder
        up1 = self.deconv1(bottleneck, enc_out4)
        up2 = self.deconv2(up1, enc_out3)
        up3 = self.deconv3(up2, enc_out2)
        up4 = self.deconv4(up3, enc_out1)
        # Final convolution
        out = self.out(up4)

        # Center crop to original size
        out = out[:, :, 4:-4, 4:-4]  # Crop back to 200x200

        # Reshape to (B, N, C)
        out = out.flatten(2).transpose(1, 2)
        return out

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
        weight_map = torch.std(t, dim=1, keepdim=True)
        l1_val = torch.mean(self.l1_loss(p, t) * (1 + self.loss_weights['std'] * (weight_map/torch.mean(weight_map))))
        losses['bev_consumer_loss_std_weighted_l1loss'] = l1_val * self.loss_weights['l1']

        # 3. MS-SSIM Loss (on Sigmoid-gated features)
        # We apply sigmoid to map unbounded features to [0, 1] for MS-SSIM stability
        p_gated = torch.sigmoid(p)
        t_gated = torch.sigmoid(t)

        # ms_ssim expects (B, C, H, W). data_range=1.0 because of sigmoid.
        # size_average=True returns a scalar loss.
        # We use 1 - ms_ssim because we want to minimize the distance
        msssim_val = 1 - ms_ssim(p_gated, t_gated, data_range=1.0, size_average=True)
        losses['bev_consumer_loss_MS_SSIM'] = msssim_val * self.loss_weights['msssim']
        
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
        loss = loss_dict['bev_consumer_loss_std_weighted_l1loss']
        
        # 3. Calculate gradients: dLoss / dPreds
        grad = torch.autograd.grad(loss, preds_for_grad)[0]
        
        # 4. Reshape to spatial (B, C, H, W) and take mean over channels
        bs, n, c = grad.shape
        grad_spatial = grad.transpose(1, 2).view(bs, c, 200, 200) # Using your bev_h/w
        grad_map = grad_spatial[0].abs().mean(dim=0).cpu().numpy() # First sample in batch

        # 5. Plot with Robust Scaling (to avoid the "one color" issue)
        plt.figure(figsize=(6, 6))
        v_min, v_max = np.percentile(grad_map, [2, 98]) # Clip outliers
        
        plt.imshow(grad_map, cmap='magma', vmin=v_min, vmax=v_max)
        plt.colorbar(label="Gradient Intensity")
        plt.title(f"Loss Gradient Map (Spatial Influence)\nEpoch {epoch}")
        
        plt.savefig(f"spatial_influence_ep{epoch}_b{batch_idx}.png")
        plt.close()