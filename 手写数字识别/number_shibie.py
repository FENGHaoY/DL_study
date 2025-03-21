from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from pandas import read_csv
from matplotlib import pyplot as plt
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
from numpy import vstack
from numpy import argmax
import torch
class CNN(Module):
    def __init__(self,n_channels):
        super(CNN,self).__init__()
        #hidden_layer1:
        self.hidden1 = Conv2d(n_channels, 32, (3,3)) #卷积的计算公式 H = H - K + 1 / S = 28-3+1=26/1 = 26
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2,2), stride=(2,2))#池化 H = H - K / S + 1 = 26 - 2 = 24 / 2 = 12 + 1 = 13 

        #hidden_layer2:
        self.hidden2 = Conv2d(32, 32, (3,3)) #11
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2,2), stride=(2,2)) #11-2 = 9/2 = 4 + 1 = 5

        #全连接层
        self.hidden3 = Linear(32*5*5, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()

        #输出层
        self.hidden4 = Linear(100,10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1) # output.shape 一般是 torch.Size([batch_size, num_classes])
        # dim=1 指定了 在哪个维度上执行 Softmax，通常是 分类的维度（通道维度）

    def forward(self,X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)

        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)

        X = X.view(-1, 5*5*32)
        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        return X
    
def prepare_data(path):
    #normalization
    trans = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])
    #加载数据集
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False,download=True, transform=trans)

    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl,test_dl

def train_model(train_dl, model,epochs=10):
    #记录损失
    losses = []
    #def 优化器
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    #epoch
    
    for epoch in range (epochs):
        running_loss = 0.0
        for i,(inputs,targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # 记录损失
    
    # 计算平均损失
        avg_loss = running_loss / len(train_dl)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i,(inputs,targets) in enumerate(test_dl):
        yhat = model(inputs) #yhat 是一个和计算图关联的张量，它保存着梯度信息。
        #convert to Numpy type
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #yhat 找出置信度最大的确定答案
        yhat = argmax(yhat, axis=1)#(batch_size, 10) argmax(yhat, axis = 1) 可以找出每个样本预测概率最大的类别索引

        yhat = yhat.reshape((len(yhat),1))
        actual = actual.reshape((len(actual),1))

        predictions.append(yhat)
        actuals.append(actual)
    #批次合并
    predictions,actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals,predictions)
    return acc

if __name__ == "__main__": #保护训练代码 防止在导入训练好的模型时候又开始训练了
    path = './data'
    train_dl,test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    model = CNN(1)
    train_model(train_dl,model,epochs=10)
    acc = evaluate_model(test_dl,model)
    print('Accuracy: %.3f' % acc)

    torch.save(model.state_dict(), 'mnist_cnn_cpu.pth')
    print("模型已保存！")