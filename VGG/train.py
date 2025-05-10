import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
from torch.utils import data as Data
from model import VGG
import copy
import pandas as pd
import matplotlib.pyplot as plt
def prepare_data():
    train_data = FashionMNIST('./data',train=True,transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download = True)
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dl = Data.DataLoader(train_data,batch_size=12,shuffle=True)
    val_dl = Data.DataLoader(val_data,batch_size=12,shuffle=True)
    return train_dl, val_dl


def train(model, train_dl, val_dl, epochs):
    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    #当前模型的参数：
    best_weights = copy.deepcopy(model.state_dict())
    #初始化参数 best_acc, train_loss[],val_loss[],train_acc[],val_acc[]
    best_acc = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        #训练：
        train_running_loss = 0.0
        train_running_acc = 0.0
        val_running_loss =0.0
        val_running_acc = 0.0
        train_num = 0
        val_num = 0
        print(f"Epoch: {epoch + 1} / {epochs}")
        print("-"*10)
        #训练
        for step,(x,y) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            model.train()
            output = model(x)
            res = torch.argmax(output, dim=1)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * x.size(0)
            train_running_acc += torch.sum(res == y.detach())#加速操作 把y的计算图分离
            train_num += x.size(0)
        #每一轮训练完成接着进行验证：（简单来说就是只求loss和acc 不更新参数）
        for step,(x,y) in enumerate(val_dl):
            x = x.to(device)
            y = y.to(device)
            model.eval()
            output = model(x)
            res = torch.argmax(output, dim=1)
            loss = criterion(output, y)
            val_running_loss += loss.item() * x.size(0)
            val_running_acc += torch.sum(res == y.detach())
            val_num += x.size(0)
        
        train_loss.append(train_running_loss / train_num)
        train_acc.append(train_running_acc / train_num)
        val_loss.append(val_running_loss / val_num)
        val_acc.append(val_running_acc / val_num)
        print(f"Train loss:{train_loss[-1]:.4f} Train acc: {train_acc[-1]:.4f}")
        print(f"Val loss:{val_loss[-1]:.4f} Val acc: {val_acc[-1]:.4f}")
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            best_weights=copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_weights)
    torch.save(best_weights,f'./LeNet5-{epochs}.pth')
    train_df = pd.DataFrame(data={
        'epochs':range(1,epochs+1),
        'train_loss':train_loss,
        'val_loss':val_loss,
        'train_acc':train_acc,
        'val_acc':val_acc
    })
    return train_df


def plot_acc_loss(train_df):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_df['epochs'], train_df['train_loss'], 'ro-', label='train_loss')
    plt.plot(train_df['epochs'], train_df['val_loss'], 'bs-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1,2,2)
    plt.plot(train_df['epochs'], train_df['train_acc'], 'ro-', label='train_acc')
    plt.plot(train_df['epochs'], train_df['val_acc'], 'bs-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig('./image/train.png')
    plt.show()

if __name__ == '__main__':
    model = VGG()
    train_dl,val_dl = prepare_data()
    epochs = 20
    train_df = train(model,train_dl,val_dl,epochs)
    plot_acc_loss(train_df)







            


            
            



