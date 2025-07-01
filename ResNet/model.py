import torch
import torch.nn as nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, input, mid, use_1conv = False, strides = 1):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input, out_channels=mid, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=mid, out_channels= mid, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.bn2 = nn.BatchNorm2d(mid)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input, out_channels=mid, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.ReLU(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = self.conv3(x)       
        y = self.ReLU(y + x)
        return y

class ResNet18(nn.Module):
    def __init__(self, Residual):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv = False, strides = 1),
            Residual(64, 64, use_1conv = False, strides = 1)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv = True, strides = 2),
            Residual(128, 128, use_1conv = False, strides = 1)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv = True, strides = 2),
            Residual(256, 256, use_1conv = False, strides = 1)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv = True, strides = 2),
            Residual(512, 512, use_1conv = False, strides = 1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return(x)


'''
Total params: 11,178,378
Trainable params: 11,178,378
'''
if __name__ == "__main__":
    device = torch.device('cuda')
    model = ResNet18(Residual).to(device)
    print(summary(model=model, input_size=(1,227,227)))