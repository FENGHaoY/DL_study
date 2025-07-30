import torch
import torch.nn as nn
import torch.nn.functional as F
from CVAE import CVAE
from torch.utils.data import DataLoader
from torch import optim as opt
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm 
from matplotlib import pyplot as plt
device = torch.device('cuda')
criterion = nn.MSELoss(reduction='sum')
batch_size = 256
lr = 0.002
epochs = 100
latent_dim = 128
n_classes = 10
def get_one_hot(label, n_classes, device):
    return torch.nn.functional.one_hot(label, n_classes).float().to(device)

def get_loss(res, image, mean, logvar):
    return criterion(res,image) + torch.sum((torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1)) / 2
def get_data():
    data = DataLoader(dataset=MNIST('./data', train=True, transform=transforms.ToTensor()
                                       ,download=True), batch_size=batch_size, shuffle=True)
    return data
def train():
    data = get_data()
    model = CVAE(label_dim = 10, input_dim=28*28, hidden_dim=256, latent_dim=latent_dim).to(device)
    optimizer = opt.Adam(params=model.parameters(), lr=lr)
    loss_list = []
    fixed_labels = torch.tensor([i for i in range(10)]).to(device)#[0-10]张量
    fixed_labels_oh = get_one_hot(fixed_labels, n_classes, device)#将固定标签进行one-hot
    for epoch in range(epochs):
        len_data = len(data)
        tmp_loss = 0.0
        cur_size = 0
        print(f"Epoch:{epoch}/{epochs}:")
        for image, label in tqdm(data):
            cur_size = len(image)
            image = image.to(device)
            label = label.to(device)
            image = image.reshape(-1,28*28)
            label = get_one_hot(label, n_classes, device=device)
            res, mean, logvar = model(image, label)
            loss = get_loss(res, image, mean, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_loss += loss.item()
        
        loss_list.append(tmp_loss / len_data / cur_size)
        print(f"loss:{loss_list[-1]}")
        if (epoch + 1) % 10 ==0 or epoch ==0:
            model.eval()
            noise = torch.randn(size=(10,latent_dim)).to(device)
            #每10个epoch展示固定标签的效果
            res = model.gen(znoise=noise, label=fixed_labels_oh).detach().cpu().numpy()
            res = res.reshape(-1, 28, 28)
            for i in range(10):
                plt.subplot(2,5,i+1)
                plt.imshow(res[i], cmap='gray')  # 绘制图像，使用灰度图
                plt.axis('off')  # 关闭坐标轴，更美观
            
            plt.savefig(f"./runs/CVAE/CVAE-res-{epoch + 1}.png")
            plt.show()
            
    torch.save(model.state_dict(), f"./runs/param/cvae-{epochs}.pth")
    print(f"模型已经保存")
    return loss_list

def show_res(loss_list: list):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label= "CVAE_loss")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('CVAE Losses')
    plt.savefig(f"./cvae_loss-{epochs}.png")
    print(f"loss可视化完成")
    
if __name__ == "__main__":
    loss_list = train()
    show_res(loss_list)
    