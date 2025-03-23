import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split, DataLoader, Dataset
from torch.nn import ReLU, Linear, Sigmoid, Module, BCELoss
from torch.optim import SGD
from torch import Tensor
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda')
class CSVDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        df = pd.read_csv(path, header=None)
        self.X = df.values[:, :-1]
        self.Y = df.values[:, -1]
        self.X = self.X.astype('float32')
        #Y转换
        self.Y = LabelEncoder().fit_transform(self.Y)
        self.Y = self.Y.astype('float32')
        self.Y = self.Y.reshape((len(self.Y), 1))
    
    #定义获取数据长度的方法
    # 定义获得数据集长度的方法
    def __len__(self):
        return len(self.X)
 
    # 定义获得某一行数据的方法
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]
    
    def get_split(self, n_test = 0.33):
        test_size = round(len(self.X) * n_test)
        train_size = len(self.X) - test_size
        return(random_split(self, [train_size, test_size]))

class MLP(Module):
    def __init__(self, inputf):
        super().__init__()
        self.hidden1 = Linear(inputf, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(8,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        return X

def prepare_data(path):
    dataset = CSVDataset(path)
    print(f"length of data:{dataset.__len__()}")
    train, test = dataset.get_split()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model,epochs= 100):
    #定义优化器
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = BCELoss()
    losses = []
    model.to(device)
    #枚举epoch
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_dl)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    plt.plot(range(1,epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss Over Time")
    plt.show()

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    model.to(device)
    model.eval()  # 进入评估模式 闭 dropout 和 batch normalization
    for i, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs) # model(inputs) 返回的是 Tensor，形状是 (batch_size, 1)，但它存储在 GPU 或 PyTorch 计算图中
        yhat = yhat.detach().cpu().numpy()#detach()消除计算图
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1)) #DataLoader 可能会自动降维，变成 (batch_size,)，需要 reshape((batch_size, 1))
        yhat = yhat.round()
        predictions.append(yhat)
        actuals.append(actual)
    
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc

def predict(row, model):
    row = Tensor([row]).to(device)
    yhat = model(row)
    yhat = yhat.detach().cpu().numpy()
    return yhat

if __name__ == '__main__':
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
    train_dl, test_dl = prepare_data(path)
    model = MLP(34)
    train_model(train_dl,model)
    acc = evaluate_model(test_dl,model)
    print(f"Accuracy:{acc:.3f}")
    row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
    yhat = predict(row, model)
    print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
    torch.save(model.state_dict(), "MLP_gpu.pth")
    print("模型已经保存")
