import torch
import numpy as np
import pandas as pd
from torch.nn import ReLU, Softmax, Linear, Module, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD
from torch import Tensor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
import matplotlib.pyplot as plt 
device = torch.device('cuda')
#数据集定义
class CSVDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        df = read_csv(path)
        self.X = df.values[:, :-1]
        self.Y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.Y = LabelEncoder().fit_transform(self.Y)
    
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, index):
        return [self.X[index],self.Y[index]]
    
    def get_splits(self, n_test= 0.33):
        test_size = round(len(self.X) * n_test)
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])
    
class MLP(Module):
    def __init__(self, input_f):
        super().__init__()
        self.hidden1 = Linear(input_f, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(10,8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(8,3)
        xavier_uniform_(self.hidden3.weight)
        self.hidden3.bias.data.fill_(0)
        #self.act3 = Softmax(dim = 1)
    
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
       # X = self.act3(X)

        return X

def prepare_data(path):
    dataset = CSVDataset(path)
    print(f"The lenth of data: {dataset.__len__()}")
    train,test = dataset.get_splits()
    train_dl  = DataLoader(train,batch_size=8,shuffle=True)
    test_dl = DataLoader(test, batch_size=1024,shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model,epochs=200):
    losses = []
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01,momentum=0.9)
    model.to(device)
    model.train()
    for epoch in range (epochs):
        avg_loss = 0.0
        for i,(inputs,targets) in enumerate(train_dl):
            inputs,targets = inputs.to(device), targets.to(device)
            targets = targets.long()
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat,targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        avg_loss = avg_loss / (len(train_dl))
        losses.append(avg_loss)
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Trainingh loss over time")
    plt.show()

def evaluate_model(test_dl, model):
    predictions,actuals = list(), list()
    model.to(device)
    model.eval()  # 进入评估模式
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dl):
            inputs,targets = inputs.to(device), targets.to(device)
            yhat = model(inputs)
            #转换类型
            yhat = yhat.detach().cpu().numpy()
            actual = targets.cpu().numpy()
            yhat = np.argmax(yhat,axis=1)

            actual = actual.reshape((len(actual),1))
            yhat = yhat.reshape((len(yhat),1))

            predictions.append(yhat)
            actuals.append(actual)
    
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc

def predict(row, model):
    model.eval()  # 设置为评估模式
    row = Tensor([row]).to(device)
    yhat = model(row)
    yhat = yhat.detach().cpu().numpy()
    return yhat

if __name__ == "__main__":
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    model = MLP(4)
    train_model(train_dl,model)
    acc = evaluate_model(test_dl,model)
    print(f"Accuracy: {acc:.3f}")
    row = [5.1,3.5,1.4,0.2]
    yhat = predict(row, model)
    print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))
    torch.save(model.state_dict(),"MLP_more.pth")
    print("模型已经保存")