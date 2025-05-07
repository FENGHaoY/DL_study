import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        #input 1*28*28
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) 
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)#6*14*14
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)#16*5*5
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=400, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)

        x = self.sigmoid(self.c3(x))
        x = self.s4(x)

        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


        

