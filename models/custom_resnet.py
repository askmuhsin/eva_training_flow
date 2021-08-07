import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMaxBR(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU() 
        )
    
    def forward(self, x):
        return self.ops(x)
        

class TwiceConv(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(
                in_channels=channel, out_channels=channel,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=channel, out_channels=channel,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channel),
            nn.ReLU() 
        )
    
    def forward(self, x):
        return self.ops(x)
        

class CustomResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer_1_1 = ConvMaxBR(64, 128)
        self.layer_1_res = TwiceConv(128)
        
        self.layer_2 = ConvMaxBR(128, 256)
        
        self.layer_3_1 = ConvMaxBR(256, 512)
        self.layer_3_res = TwiceConv(512)
        
        self.maxpool_out = nn.MaxPool2d(kernel_size=(4, 4))
        self.linear = nn.Linear(512, 10)
        
    
    def forward(self, x):
        # Prep
        x = self.prep_layer(x)
        
        # Layer 1
        x = self.layer_1_1(x)
        r1 = self.layer_1_res(x)
        x = x + r1
        
        # Layer 2
        x = self.layer_2(x)
        
        # Layer 3
        x = self.layer_3_1(x)
        r2 = self.layer_3_res(x)
        x = x + r2
        
        # out
        x = self.maxpool_out(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return F.log_softmax(x, dim=-1)


