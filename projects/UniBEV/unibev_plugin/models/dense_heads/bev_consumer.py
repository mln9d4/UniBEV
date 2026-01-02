from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn

@HEADS.register_module()
class MyBEVConsumer(BaseModule):
    def __init__(self, in_channels=256, hidden=256, out_channels=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, bev_embed):
        # bev_embed: (bs, bev_h * bev_w, C)
        return self.mlp(bev_embed)  # shape (bs, bev_h * bev_w, out_channels)

    def loss(self, preds, targets):
        # preds and targets should have same shape: (bs, bev_h * bev_w, out_channels)
        return {'loss_bev_consumer': self.loss_fn(preds, targets)}

@HEADS.register_module()
class UnetFeatureMapping4layers(BaseModule):
    #V2 including batchnorm layers
    def __init__(self, input_channels=256, output_channels=256, loss_fn=nn.L1Loss(), bev_h=200, bev_w=200):
        super().__init__()

        # Define loss function
        self.loss_fn = loss_fn
        self.bev_h = bev_h
        self.bev_w = bev_w
        
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
        bs = x.shape[0]
        x = x.permute(0, 2, 1).reshape(bs, -1, self.bev_h, self.bev_w)
        
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
        
        # Reshape back to (bs, bev_h * bev_w, output_channels)
        x = x.reshape(bs, -1, self.bev_h * self.bev_w).permute(0, 2, 1)
        
        return x
        
    def loss(self, preds, targets):
        # preds and targets should have same shape: (bs, bev_h * bev_w, out_channels)
        return {'loss_bev_consumer': self.loss_fn(preds, targets)}
    


@HEADS.register_module()
class UnetFeatureMapping3layers(BaseModule):
    def __init__(self, input_channels=256, output_channels=256, loss_fn=nn.L1Loss(), bev_h=200, bev_w=200):
        super().__init__()

        # Define loss function
        self.loss_fn = loss_fn
        self.bev_h = bev_h
        self.bev_w = bev_w
        
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
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_sizes[1], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_sizes[2], self.channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (expanding path)
        
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
        bs = x.shape[0]
        x = x.permute(0, 2, 1).reshape(bs, -1, self.bev_h, self.bev_w)
        
        # Encoder path - save skip connections
        enc1_out = self.enc1(x)          
        x = self.pool1(enc1_out)          
        
        enc2_out = self.enc2(x)           
        x = self.pool2(enc2_out)          
        
        # Bottleneck
        x = self.bottleneck(x)            
        
        # Decoder path - concatenate skip connections                       
        
        x = self.upconv2(x)               
        x = torch.cat([x, enc2_out], dim=1)  
        x = self.dec2(x)                  
        
        x = self.upconv1(x)               
        x = torch.cat([x, enc1_out], dim=1)  
        x = self.dec1(x)                  
        
        # Final layer
        x = self.final(x)
        
        # Reshape back to (bs, bev_h * bev_w, output_channels)
        x = x.reshape(bs, -1, self.bev_h * self.bev_w).permute(0, 2, 1)
        
        return x
        
    def loss(self, preds, targets):
        # preds and targets should have same shape: (bs, bev_h * bev_w, out_channels)
        return {'loss_bev_consumer': self.loss_fn(preds, targets)}