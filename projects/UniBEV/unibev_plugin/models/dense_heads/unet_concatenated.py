from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

@HEADS.register_module()
class UnetConcatenated(BaseModule):
    """
    Flexible U-Net with 4 layers and CONCATENATED skip connections.
    Uses only L1 loss. This is a baseline model for comparison.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [256, 512, 1024, 2048]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200, 
                 loss_weights=dict(l1=1.0, msssim=0.0)):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        # Zhao paper says 0.16 for l1 and 0.84 for ms-ssim
        self.loss_weights = loss_weights  # Only L1
        self.l1_loss = nn.L1Loss()

        # self.input_pca = nn.Sequential(
        #     nn.Conv2d(input_channels, channel_sizes[0], kernel_size=1),
        #     nn.GroupNorm(32, channel_sizes[0])
        # )
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder blocks (expanding path) - Original U-Net style concatenation
        self.upconv3 = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channel_sizes[2] * 2, channel_sizes[2], kernel_size=3, padding=1), # Input channels are doubled
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=3, padding=1), # Input channels are doubled
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=3, padding=1), # Input channels are doubled
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        
        self.final = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)
    
    def forward(self, x):
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # x = self.input_pca(x)
        enc1_out = self.enc1(x) # out:(200,200,256)
        x = self.pool1(enc1_out) # out:(100,100,256)
        
        enc2_out = self.enc2(x) # out:(100,100,512)
        x = self.pool2(enc2_out) # out:(50,50,512)
        
        enc3_out = self.enc3(x) # out:(50,50,1024)
        x = self.pool3(enc3_out) # out:(25,25,1024)
        
        x = self.bottleneck(x) # out:(25,25,2048)
        
        x = self.upconv3(x) # out:(50,50,1024)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x) # out:(100,100,512)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x) # out:(200,200,256)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x) # out:(200,200,256)
        
        x = self.final(x)
        
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
            """
            Calculates L1 and MS-SSIM loss with Sigmoid gating.
            """
            losses = {}
            
            # 1. Reshape back to (B, C, H, W) for spatial losses
            bs, n_points, c = preds.shape
            p = preds.transpose(1, 2).view(bs, c, self.bev_h, self.bev_w)
            t = targets.transpose(1, 2).view(bs, c, self.bev_h, self.bev_w)

            # 2. Standard L1 Loss (on raw features)
            l1_val = self.l1_loss(p, t)
            losses['bev_consumer_loss_L1'] = l1_val * self.loss_weights['l1']

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