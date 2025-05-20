import torch
import torch.nn as nn
from torchsummary import summary

class inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.ReLU = nn.ReLU()
        # 路线一 单一卷积层
        self.p1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        #路线二 1*1 + 3*3
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        #路线三 1*1 +5*5
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        #路线四 3*3 maxpool + 1*1
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)
    
    def forward(self,x):
        p1 = self.ReLU(self.p1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # (batch, channel, H, W)


class GoogLeNet(nn.Module):
    def __init__(self,inception):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b3 = nn.Sequential(
            inception(192, 64, (96, 128), (16, 32), 32),
            inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b4 = nn.Sequential(
            inception(480, 192, (96, 208),(16, 48), 64),
            inception(512, 160, (112, 224), (24, 64), 64),
            inception(512, 128, (128, 256), (24, 64), 64),
            inception(512, 112, (128, 288), (32, 64), 64),
            inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            inception(832, 256, (160, 320), (32, 128), 128),
            inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda')
    model = GoogLeNet(inception).to(device)
    print(summary(model, (1,224,224))) 
    