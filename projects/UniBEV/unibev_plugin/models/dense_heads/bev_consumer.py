from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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
        
        self.loss_fn = nn.MSELoss()
        
        # Define block channel sizes
        self.channel_sizes = [512, 1024, 2048, 4096]


        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, self.channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[3], self.channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (expanding path)
        self.upconv3 = nn.ConvTranspose2d(self.channel_sizes[3], self.channel_sizes[2], kernel_size=2, stride=2)  # 25->50
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[3], self.channel_sizes[2], kernel_size=3, padding=1),  # 4096 = 2048 + 2048
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(self.channel_sizes[2], self.channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[1], kernel_size=3, padding=1),  # 2048 = 1024 + 1024
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[0], kernel_size=3, padding=1),  # 1024 = 512 + 512
            nn.BatchNorm2d(self.channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[0], self.channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[0]),    
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
        
        losses['bev_consumer_loss_L1Loss'] = self.loss_fn(preds, targets)
        
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
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=3, padding=1),
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
        # self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        # self.weight_1 = 1.0  # Weight for L1Loss
        # self.weight_2 = 0.1  # Weight for CosineEmbeddingLoss
        
        # Encoder blocks (contracting path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
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
            nn.BatchNorm2d(channel_sizes[2]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        # After concat: channel_sizes[1] + channel_sizes[1] = 2 * channel_sizes[1]
        # 1x1 conv to reduce to channel_sizes[1]
        self.skip_conv2 = nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        # After concat: channel_sizes[0] + channel_sizes[0] = 2 * channel_sizes[0]
        # 1x1 conv to reduce to channel_sizes[0]
        self.skip_conv1 = nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
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
        
        # CosineEmbeddingLoss with weight 0.5 and target=1 (similarity)
        # Flatten to (batch_size, embedding_dim)
        # p = preds.reshape(preds.size(0), -1)
        # t = targets.reshape(targets.size(0), -1)
        # y = torch.ones(p.size(0), device=p.device)  # target=1 for similarity
        # cosine_loss_value = self.cosine_loss(p, t, y)
        # losses['bev_consumer_loss_CosineEmbeddingLoss'] = self.weight_2 * cosine_loss_value

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
        
        # CosineEmbeddingLoss with weight 0.5 and target=1 (similarity)
        # Flatten to (batch_size, embedding_dim)
        # p = preds.reshape(preds.size(0), -1)
        # t = targets.reshape(targets.size(0), -1)
        # y = torch.ones(p.size(0), device=p.device)  # target=1 for similarity
        # cosine_loss_value = self.cosine_loss(p, t, y)
        # losses['bev_consumer_loss_CosineEmbeddingLoss'] = self.weight_2 * cosine_loss_value

        return losses


@HEADS.register_module()
class FlexibleUNet(BaseModule):
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
            nn.GroupNorm(32, channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200->100
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100->50
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 50->25
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[3], channel_sizes[3], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[3]),
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
            nn.GroupNorm(32, channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[2], channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 2
        self.upconv2 = nn.ConvTranspose2d(channel_sizes[2], channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        # After concat: channel_sizes[1] + channel_sizes[1] = 2 * channel_sizes[1]
        # 1x1 conv to reduce to channel_sizes[1]
        self.skip_conv2 = nn.Conv2d(channel_sizes[1] * 2, channel_sizes[1], kernel_size=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[1], channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 1
        self.upconv1 = nn.ConvTranspose2d(channel_sizes[1], channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        # After concat: channel_sizes[0] + channel_sizes[0] = 2 * channel_sizes[0]
        # 1x1 conv to reduce to channel_sizes[0]
        self.skip_conv1 = nn.Conv2d(channel_sizes[0] * 2, channel_sizes[0], kernel_size=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_sizes[0], channel_sizes[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, channel_sizes[0]),
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
        
        # CosineEmbeddingLoss with weight 0.5 and target=1 (similarity)
        # Flatten to (batch_size, embedding_dim)
        # p = preds.reshape(preds.size(0), -1)
        # t = targets.reshape(targets.size(0), -1)
        # y = torch.ones(p.size(0), device=p.device)  # target=1 for similarity
        # cosine_loss_value = self.cosine_loss(p, t, y)
        # losses['bev_consumer_loss_CosineEmbeddingLoss'] = self.weight_2 * cosine_loss_value

        return losses

@HEADS.register_module()
class UnetFeatureMapping4layersV2(BaseModule):
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
        self.upconv3 = nn.ConvTranspose2d(self.channel_sizes[3], self.channel_sizes[2], kernel_size=2, stride=2)  # 25->50
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[3], self.channel_sizes[2], kernel_size=3, padding=1),  # 4096 = 2048 + 2048
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(self.channel_sizes[2], self.channel_sizes[1], kernel_size=2, stride=2)  # 50->100
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[1], kernel_size=3, padding=1),  # 2048 = 1024 + 1024
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.channel_sizes[1]),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[0], kernel_size=2, stride=2)  # 100->200
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[0], kernel_size=3, padding=1),  # 1024 = 512 + 512
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