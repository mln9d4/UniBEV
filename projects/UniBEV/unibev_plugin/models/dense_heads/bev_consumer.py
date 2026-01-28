from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import os

@HEADS.register_module()
class DebugOverfitHead(BaseModule):
    def __init__(self, input_channels=256, hidden=256, output_channels=256, bev_h=200, bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, output_channels, kernel_size=1),
        )


        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (bs, H*W, C)
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        x = self.net(x)
        x = x.flatten(2).transpose(1, 2)  # back to (bs, H*W, C)
        return x

    def loss(self, preds, targets):
        return dict(loss_debug=self.loss_fn(preds, targets))

@HEADS.register_module()
class MyBEVConsumer(BaseModule):
    def __init__(self, input_channels=256, hidden=256, output_channels=256, bev_h=200, bev_w=200):
        super().__init__()

        
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output_channels),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, bev_embed):
        # bev_embed: (bs, bev_h * bev_w, C)
        return self.mlp(bev_embed)  # shape (bs, bev_h * bev_w, out_channels)

    def loss(self, preds, targets):
        # preds and targets should have same shape: (bs, bev_h * bev_w, out_channels)
        return {'bev_consumer_loss_L1Loss': self.loss_fn(preds, targets)}


@HEADS.register_module()
class UnetFeatureMapping4layers(BaseModule):
    #V2 including batchnorm layers
    def __init__(self, input_channels=256, output_channels=256, bev_h=200, bev_w=200):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        

        self.loss_fn = nn.L1Loss()
        
        # Define block channel sizes
        self.channel_sizes = [256, 512, 1024, 2048]


        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, self.channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[3], self.channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (expanding path)
        self.upconv3 = nn.ConvTranspose2d(self.channel_sizes[3], self.channel_sizes[2], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(self.channel_sizes[2], self.channel_sizes[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[0]),    
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(self.channel_sizes[0], output_channels, kernel_size=1)

        
    def forward(self, x):
        # Input shape: (bs, bev_h * bev_w, C)
        # Reshape to (bs, C, bev_h, bev_w) for Conv2d
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # Encoder path - save skip connections
        enc1_out = self.enc1(x)          
        x = self.pool1(enc1_out)          
        
        enc2_out = self.enc2(x)           
        x = self.pool2(enc2_out)          
        
        enc3_out = self.enc3(x)           
        x = self.pool3(enc3_out)          
        
        # Bottleneck
        x = self.bottleneck(x)            
        
        # Decoder path - concatenate skip connections        
        x = self.upconv3(x)               
        x = torch.cat([x, enc3_out], dim=1)  
        x = self.dec3(x)                  
        
        x = self.upconv2(x)               
        x = torch.cat([x, enc2_out], dim=1)  
        x = self.dec2(x)                  
        
        x = self.upconv1(x)               
        x = torch.cat([x, enc1_out], dim=1)  
        x = self.dec1(x)                  
        
        # Final layer
        x = self.final(x)
        
        # Back to (bs, bev_h * bev_w, output_channels)
        x = x.flatten(2).transpose(1, 2)
        return x
        
    def loss(self, preds, targets):
        """Compute weighted loss.
        
        Args:
            preds: Predicted tensor (bs, bev_h * bev_w, out_channels)
            targets: Target tensor (same shape as preds)
        
        Returns:
            dict: Dictionary with total loss and individual loss components
        """
        losses = {}
        
        loss_l1 = self.loss_fn(preds, targets)
        losses['bev_consumer_l1_loss'] = loss_l1
        
        return losses

@HEADS.register_module()
class FlexibleUNetSiLU(BaseModule):
    """
    Flexible U-Net with 4 layers and configurable channel sizes.
    Features 1x1 convolutions after skip concatenations to reduce channels.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [512, 1024, 2048, 4096]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
        # Initialize loss functions with fixed configuration
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        # self.weight_1 = 1.0  # Weight for L1Loss
        # self.weight_2 = 0.1  # Weight for CosineEmbeddingLoss

        self.input_pca = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=1),
            nn.GroupNorm(32, channel_sizes[0])
        )
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder blocks (expanding path) with 1x1 convolutions for channel reduction
        # Decoder 3
        self.upconv3 = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=2, stride=2)  # 25->50
        # After concat: channel_sizes[2] + channel_sizes[2] = 2 * channel_sizes[2]
        # 1x1 conv to reduce to channel_sizes[2]
        self.skip_conv3 = nn.Conv2d(channel_sizes[2] * 2, channel_sizes[2], kernel_size=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        # After concat: channel_sizes[1] + channel_sizes[1] = 2 * channel_sizes[1]
        # 1x1 conv to reduce to channel_sizes[1]
        self.skip_conv2 = nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        # After concat: channel_sizes[0] + channel_sizes[0] = 2 * channel_sizes[0]
        # 1x1 conv to reduce to channel_sizes[0]
        self.skip_conv1 = nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (bs, bev_h * bev_w, C)
        
        Returns:
            Output tensor of shape (bs, bev_h * bev_w, output_channels)
        """
        # Input shape: (bs, bev_h * bev_w, C)
        # Reshape to (bs, C, bev_h, bev_w) for Conv2d
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # Encoder path - save skip connections
        x = self.input_pca(x)
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - concatenate skip connections with 1x1 conv reduction
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.skip_conv3(x)  # 1x1 conv to reduce channels
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.skip_conv2(x)  # 1x1 conv to reduce channels
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.skip_conv1(x)  # 1x1 conv to reduce channels
        x = self.dec1(x)
        
        # Final layer
        x = self.final(x)
        
        # Back to (bs, bev_h * bev_w, output_channels)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
        """
        Compute combined loss: 1.0 * L1Loss + 0.5 * CosineEmbeddingLoss.
        
        Args:
            preds: Predicted tensor (bs, bev_h * bev_w, out_channels)
            targets: Target tensor (same shape as preds)
        
        Returns:
            dict: Dictionary with loss components
        """
        losses = {}

        # L1 Loss with weight 1.0
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value

        # Cosine: compute per-pixel (over channels), then average
        # Reshape to (B, C, H, W) for consistency with eval
        # B = preds.size(0)
        # preds_2d = preds.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        # targets_2d = targets.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        
        # cosine_per_pixel = torch.nn.functional.cosine_similarity(preds_2d, targets_2d, dim=1)  # (B, H, W)
        # cosine_loss_value = 1.0 - cosine_per_pixel.mean()  # Convert similarity to loss

        # losses['bev_consumer_loss_CosineSimilarity'] = cosine_loss_value

        return losses


@HEADS.register_module()
class FlexibleUNetBatchNormSiLU(BaseModule):
    """
    Flexible U-Net with 4 layers and configurable channel sizes.
    Features 1x1 convolutions after skip concatenations to reduce channels.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [512, 1024, 2048, 4096]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
        # Initialize loss functions with fixed configuration
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        # self.weight_1 = 1.0  # Weight for L1Loss
        # self.weight_2 = 0.1  # Weight for CosineEmbeddingLoss

        self.input_pca = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=1),
            nn.GroupNorm(32, channel_sizes[0])
        )
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder blocks (expanding path) with 1x1 convolutions for channel reduction
        # Decoder 3
        self.upconv3 = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=2, stride=2)  # 25->50
        # After concat: channel_sizes[2] + channel_sizes[2] = 2 * channel_sizes[2]
        # 1x1 conv to reduce to channel_sizes[2]
        self.skip_conv3 = nn.Conv2d(channel_sizes[2] * 2, channel_sizes[2], kernel_size=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        # After concat: channel_sizes[1] + channel_sizes[1] = 2 * channel_sizes[1]
        # 1x1 conv to reduce to channel_sizes[1]
        self.skip_conv2 = nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        # After concat: channel_sizes[0] + channel_sizes[0] = 2 * channel_sizes[0]
        # 1x1 conv to reduce to channel_sizes[0]
        self.skip_conv1 = nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (bs, bev_h * bev_w, C)
        
        Returns:
            Output tensor of shape (bs, bev_h * bev_w, output_channels)
        """
        # Input shape: (bs, bev_h * bev_w, C)
        # Reshape to (bs, C, bev_h, bev_w) for Conv2d
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # Encoder path - save skip connections
        x = self.input_pca(x)
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - concatenate skip connections with 1x1 conv reduction
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.skip_conv3(x)  # 1x1 conv to reduce channels
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.skip_conv2(x)  # 1x1 conv to reduce channels
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.skip_conv1(x)  # 1x1 conv to reduce channels
        x = self.dec1(x)
        
        # Final layer
        x = self.final(x)
        
        # Back to (bs, bev_h * bev_w, output_channels)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
        """
        Compute combined loss: 1.0 * L1Loss + 0.5 * CosineEmbeddingLoss.
        
        Args:
            preds: Predicted tensor (bs, bev_h * bev_w, out_channels)
            targets: Target tensor (same shape as preds)
        
        Returns:
            dict: Dictionary with loss components
        """
        losses = {}

        # L1 Loss with weight 1.0
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value

        # Cosine: compute per-pixel (over channels), then average
        # Reshape to (B, C, H, W) for consistency with eval
        # B = preds.size(0)
        # preds_2d = preds.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        # targets_2d = targets.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        
        # cosine_per_pixel = torch.nn.functional.cosine_similarity(preds_2d, targets_2d, dim=1)  # (B, H, W)
        # cosine_loss_value = 1.0 - cosine_per_pixel.mean()  # Convert similarity to loss

        # losses['bev_consumer_loss_CosineSimilarity'] = cosine_loss_value

        return losses


@HEADS.register_module()
class FlexibleUNetBatchNorm(BaseModule):
    """
    Flexible U-Net with 4 layers and configurable channel sizes.
    Features 1x1 convolutions after skip concatenations to reduce channels.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [512, 1024, 2048, 4096]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
        # Initialize loss functions with fixed configuration
        self.l1_loss = nn.L1Loss()
        # self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        # self.weight_1 = 1.0  # Weight for L1Loss
        # self.weight_2 = 0.1  # Weight for CosineEmbeddingLoss
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (expanding path) with 1x1 convolutions for channel reduction
        # Decoder 3
        self.upconv3 = nn.ConvTranspose2d(channel_sizes[3], channel_sizes[2], kernel_size=2, stride=2)  # 25->50
        # After concat: channel_sizes[2] + channel_sizes[2] = 2 * channel_sizes[2]
        # 1x1 conv to reduce to channel_sizes[2]
        self.skip_conv3 = nn.Conv2d(channel_sizes[2] * 2, channel_sizes[2], kernel_size=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        # After concat: channel_sizes[1] + channel_sizes[1] = 2 * channel_sizes[1]
        # 1x1 conv to reduce to channel_sizes[1]
        self.skip_conv2 = nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        # After concat: channel_sizes[0] + channel_sizes[0] = 2 * channel_sizes[0]
        # 1x1 conv to reduce to channel_sizes[0]
        self.skip_conv1 = nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(channel_sizes[0], output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (bs, bev_h * bev_w, C)
        
        Returns:
            Output tensor of shape (bs, bev_h * bev_w, output_channels)
        """
        # Input shape: (bs, bev_h * bev_w, C)
        # Reshape to (bs, C, bev_h, bev_w) for Conv2d
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # Encoder path - save skip connections
        x = self.input_pca(x)
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - concatenate skip connections with 1x1 conv reduction
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.skip_conv3(x)  # 1x1 conv to reduce channels
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.skip_conv2(x)  # 1x1 conv to reduce channels
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.skip_conv1(x)  # 1x1 conv to reduce channels
        x = self.dec1(x)
        
        # Final layer
        x = self.final(x)
        
        # Back to (bs, bev_h * bev_w, output_channels)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
        """
        Compute combined loss: 1.0 * L1Loss + 0.5 * CosineEmbeddingLoss.
        
        Args:
            preds: Predicted tensor (bs, bev_h * bev_w, out_channels)
            targets: Target tensor (same shape as preds)
        
        Returns:
            dict: Dictionary with loss components
        """
        losses = {}

        # L1 Loss with weight 1.0
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value

        # Cosine: compute per-pixel (over channels), then average
        # Reshape to (B, C, H, W) for consistency with eval
        # B = preds.size(0)
        # preds_2d = preds.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        # targets_2d = targets.permute(0, 2, 1).view(B, -1, self.bev_h, self.bev_w)
        
        # cosine_per_pixel = torch.nn.functional.cosine_similarity(preds_2d, targets_2d, dim=1)  # (B, H, W)
        # cosine_loss_value = 1.0 - cosine_per_pixel.mean()  # Convert similarity to loss

        # losses['bev_consumer_loss_CosineSimilarity'] = cosine_loss_value

        return losses

@HEADS.register_module()
class UnetAdditive(BaseModule):
    """
    Flexible U-Net with 4 layers and ADDITIVE skip connections.
    Includes a composite loss function (L1 + Cosine + TV).
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')

        self.input_pca = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=1),
            nn.GroupNorm(32, channel_sizes[0])
        )
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
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
        
        # Decoder blocks (expanding path)
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

    def total_variation_loss(self, img, weight=1.0):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

    def forward(self, x):
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        # Encoder path
        x = self.input_pca(x)
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with ADDITIVE skip connections
        x = self.upconv3(x)
        x = x + enc3_out  # Additive skip connection
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = x + enc2_out  # Additive skip connection
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = x + enc1_out  # Additive skip connection
        x = self.dec1(x)
        
        # Final layer
        x = self.final(x)
        
        # Keep the 2D shape for loss calculation, then flatten
        flat_preds = x.flatten(2).transpose(1, 2)
        
        # The model must return a tuple: (flattened_preds, 2d_preds)
        # The training harness might only expect one output. This will be handled
        # by a small change in the training code where the model is called.
        return flat_preds

    def loss(self, preds, targets):
        losses = {}
        
        # L1 Loss
        loss_l1 = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1'] = loss_l1
        
        # Cosine Embedding Loss
        # flat_preds = preds.contiguous().view(-1, self.channel_sizes[0])
        # flat_targets = targets.contiguous().view(-1, self.channel_sizes[0])
        # y = torch.ones(flat_preds.size(0)).to(preds.device)
        # loss_cosine = self.cosine_loss(flat_preds, flat_targets, y)
        # losses['bev_consumer_loss_Cosine'] = loss_cosine

        # # Total Variation Loss
        # loss_tv = self.total_variation_loss(preds_2d)
        # losses['bev_consumer_loss_TV'] = loss_tv

        # # Combined Loss (tune these weights)
        # total_loss = loss_l1 + 0.1 * loss_cosine + 0.01 * loss_tv
        # losses['bev_consumer_loss_Total'] = total_loss

        return losses

@HEADS.register_module()
class UnetConcatenated(BaseModule):
    """
    Flexible U-Net with 4 layers and CONCATENATED skip connections.
    Uses only L1 loss. This is a baseline model for comparison.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [256, 256, 256, 256]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
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
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        x = self.bottleneck(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x)
        
        x = self.final(x)
        
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
        
        losses = {}
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value
        return losses
    
@HEADS.register_module()
class UnetConcatenatedPCA(BaseModule):
    """
    Flexible U-Net with 4 layers and CONCATENATED skip connections.
    Uses only L1 loss. This is a baseline model for comparison.
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        channel_sizes (list): Channel sizes for each layer, length must be 4
                             e.g., [256, 256, 256, 256]
        bev_h (int): Height of BEV grid
        bev_w (int): Width of BEV grid
    """
    def __init__(self, input_channels=256, output_channels=256, 
                 channel_sizes=[256, 256, 256, 256],
                 bev_h=200, bev_w=200):
        super().__init__()
        
        assert len(channel_sizes) == 4, "channel_sizes must have exactly 4 elements"
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channel_sizes = channel_sizes
        
        self.l1_loss = nn.L1Loss()

        self.input_pca = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=1),
            nn.GroupNorm(32, channel_sizes[0])
        )
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
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
        
        x = self.input_pca(x)
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)
        
        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)
        
        enc3_out = self.enc3(x)
        x = self.pool3(enc3_out)
        
        x = self.bottleneck(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3_out], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x)
        
        x = self.final(x)
        
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def loss(self, preds, targets):
        
        losses = {}
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value
        return losses

@HEADS.register_module()
class SimpleBEVConsumer(BaseModule):
    """
    A very simple consumer model for debugging.
    It consists of a few convolutional layers.
    """
    def __init__(self, input_channels=256, output_channels=256, bev_h=200, bev_w=200, channel_sizes=[256, 256, 256, 256]):
        super().__init__()
        self.bev_h = bev_h
        self.channel_sizes = channel_sizes
        self.bev_w = bev_w
        self.input = None
        self.l1_loss = nn.L1Loss()
        self.counter = 0
        self.debug_dump_done = False

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=1)
        )

    def forward(self, x):
        self.input = x
        bs, v, c_in = x.shape
        assert v == self.bev_h * self.bev_w, f"V={v} must equal bev_h*bev_w={self.bev_h*self.bev_w}"
        x = x.permute(0, 2, 1).contiguous().view(bs, c_in, self.bev_h, self.bev_w)
        
        x = self.model(x)
        
        x = x.flatten(2).transpose(1, 2)


        # if not self.debug_dump_done:
        #     print("\n\n---!!! DEBUGGING: DUMPING FORWARD OUTPUT !!!---\n")
        #     # Reshape for visualization
        #     output_reshaped = x.detach().cpu()

        #     # Save the tensors to files
        #     dump_dir = "debug_dump"
        #     os.makedirs(dump_dir, exist_ok=True)
        #     torch.save(output_reshaped, os.path.join(dump_dir, f"debug_preds_at_loss{self.counter}.pt"))
        #     torch.save(self.input.detach().cpu(), os.path.join(dump_dir, f"debug_input_at_loss{self.counter}.pt"))
        #     print(f"Saved 'forward output' to '{dump_dir}' directory.")
        #     print("---!!! DEBUGGING: DUMP COMPLETE !!!---\n\n")
        #     self.counter += 1
        #     if self.counter >= 3:
        #         self.debug_dump_done = True
        return x

    def loss(self, preds, targets):
                # --- DEBUGGING CODE ---
        if not self.debug_dump_done:
            print("\n\n---!!! DEBUGGING: DUMPING TENSORS !!!---\n")
            # The 'preds' tensor is the input to the loss function, which is the output of the forward pass.
            # The 'targets' tensor is the ground truth your model is being compared against.
            
            # Let's get the input to the model as well. The framework passes it as the first argument to forward.
            # To get it here, we need to find where it's stored. In UniBEV, the input to the consumer
            # is often the output of the transformer decoder. Let's just save preds and targets for now.

            # Reshape for visualization
            preds_reshaped = preds.detach().cpu()
            targets_reshaped = targets.detach().cpu()
            input_reshaped = self.input.detach().cpu()

            # Save the tensors to files
            dump_dir = "debug_dump"
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(input_reshaped, os.path.join(dump_dir, f"debug_input_at_loss{self.counter}.pt"))
            torch.save(preds_reshaped, os.path.join(dump_dir, f"debug_preds_at_loss{self.counter}.pt"))
            torch.save(targets_reshaped, os.path.join(dump_dir, f"debug_targets_at_loss{self.counter}.pt"))
            
            print(f"Saved 'preds' and 'targets' to '{dump_dir}' directory.")
            print(f"Target stats: Mean={targets.mean():.4f}, Std={targets.std():.4f}, Min={targets.min():.4f}, Max={targets.max():.4f}")
            print("---!!! DEBUGGING: DUMP COMPLETE !!!---\n\n")
            self.counter += 1
            if self.counter >= 3:
                self.debug_dump_done = True
        # --- END DEBUGGING CODE ---
        losses = {}
        l1_loss_value = self.l1_loss(preds, targets)
        losses['bev_consumer_loss_L1Loss'] = l1_loss_value
        return losses