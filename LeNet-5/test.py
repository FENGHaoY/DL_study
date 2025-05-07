import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
from torch.utils import data as Data
from model import LeNet
import copy
import pandas as pd
import matplotlib.pyplot as plt
def prepare_data():
    test_data = FashionMNIST('./data',train=False,transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]))
    test_dl = Data.DataLoader(test_data,batch_size=32,shuffle=False)#不用随机打乱
    return test_dl

def test(model,test_dl):
    device = torch.device('cuda')
    model.to(device)
    val_num = 0
    acc = 0.0
    running_acc = 0
    with torch.no_grad():
        for index,(x,y) in enumerate(test_dl):
            x = x.to(device)
            y = y.to(device)
            model.eval()
            output = model(x)
            res = torch.argmax(output,dim=1)
            running_acc += torch.sum(res==y.detach())
            val_num += x.size(0)
        
        acc = running_acc/val_num
    
    print(f"测试的准确率为：{acc:.2f}")

if __name__ == '__main__':
    leNet = LeNet()
    leNet.load_state_dict(torch.load('./LeNet5-20.pth'))
    test_dl = prepare_data()
    test(leNet, test_dl)

    #推理：
    device = torch.device('cuda')
    with torch.no_grad():
        leNet.to(device)
        for index,(x,y) in enumerate(test_dl):
            x = x.to(device)
            y = y.to(device)
            leNet.eval()
            output = leNet(x)
            res = torch.argmax(output,dim=1)
            for i in range(len(res)):
                print(f"预测值：{res[i].item()}, 实际值{y[i].item()}")
