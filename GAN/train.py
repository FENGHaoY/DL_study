import torch 
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from model import gen, disc
from torch import optim as opt
from tqdm.auto import tqdm 
from torchvision.utils import make_grid
import numpy as np
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
#超参数
batch_size = 128
noise_dim = 64
lr = 0.0001
beta_1 = 0.5
beta_2 = 0.999
epochs = 200
device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()
def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
            
def show_tensor_images(image_tensor, epoch, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"./runs/res/res-{epoch}")
    plt.show()


def get_noise(num:int, noise_dim:int, device = device):
    noise = torch.randn(size=(num, noise_dim)).to(device)
    return noise

def get_gen_loss(gen, disc, batch_size, noise_dim, criterion:nn.BCEWithLogitsLoss, device):
    noise = get_noise(batch_size, noise_dim, device)
    fake_img = gen(noise)
    fake_prd = disc(fake_img)
    gen_loss = criterion(fake_prd, torch.ones_like(fake_prd))
    return gen_loss

def get_disc_loss(gen, disc, batch_size, noise_dim, criterion, device, real):
    #real: a batch of real imgs
    #disc的目标：假的预测成假的 真的预测成真的
    noise = get_noise(batch_size, noise_dim, device)
    fake_img = gen(noise)
    fake_prd = disc(fake_img.detach())
    '''
    如果去掉 .detach()，在更新判别器的阶段（执行 disc_loss.backward() 时），
    梯度会沿着这样的路径传播：disc_loss → fake_prd → fake_img → gen 的参数
    '''
    fake_loss = criterion(fake_prd, torch.zeros_like(fake_prd))
    real_prd = disc(real)
    real_loss = criterion(real_prd, torch.ones_like(real_prd))
    disc_loss = (fake_loss + real_loss) / 2
    return (disc_loss)

def train():
    dataloader = DataLoader(MNIST('./data',train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), download=True),batch_size=batch_size,shuffle=True)
    gen_model = gen(noise_dim=noise_dim).to(device)
    gen_opt = opt.Adam(params=gen_model.parameters(),lr=lr,betas=(beta_1,beta_2))
    disc_model = disc().to(device)
    disc_opt = opt.Adam(params=disc_model.parameters(),lr=lr,betas=(beta_1,beta_2))
    gen_model.apply(weights_init)
    disc_model.apply(weights_init)
    gen_loss_list = []
    disc_loss_list = []
    for epoch in range(epochs):
        print(f"Epochs: {epoch + 1} / {epochs}")
        mean_disc_loss = 0
        mean_gen_loss = 0
        cur_len = 0
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device) #数据送上gpu
            #更新disc
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen_model, disc_model, cur_batch_size, noise_dim=noise_dim,criterion=criterion, device=device, real=real)
            disc_loss.backward()
            disc_opt.step()
            #更新gen
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen=gen_model, disc=disc_model, batch_size=cur_batch_size, noise_dim=noise_dim, criterion=criterion, device=device)
            gen_loss.backward()
            gen_opt.step()
            mean_disc_loss += disc_loss.item()
            mean_gen_loss += gen_loss.item()
            cur_len += 1
        
        mean_disc_loss = mean_disc_loss / cur_len
        mean_gen_loss = mean_gen_loss / cur_len
        cur_len = 0
        gen_loss_list.append(mean_gen_loss)
        disc_loss_list.append(mean_disc_loss)
        
        print(f"discloss:{mean_disc_loss} genloss:{mean_gen_loss}")
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_noise = get_noise(num=25, noise_dim=noise_dim, device=device)  # 生成25张图像
                fake_imgs = gen_model(fake_noise)
                show_tensor_images(real,epoch=666)
                show_tensor_images(fake_imgs, num_images=25,epoch=epoch+1)  # 调用可视化函数
    
    
    torch.save(gen_model.state_dict(), f'./runs/model_params/gen_model_params-{epochs}.pth')
    # 保存判别器模型参数
    torch.save(disc_model.state_dict(), f'./runs/model_params/disc_model_params-{epochs}.pth')
    return gen_loss_list, disc_loss_list
            
def plot_res(gen_list:list, disc_list:list):
    plt.figure(figsize=(10, 6))
    plt.plot(gen_list, label='Generator Loss')
    plt.plot(disc_list, label='Discriminator Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('GAN Losses')
    plt.savefig(f"./runs/GAN_Losses-{epochs}")
    plt.show()

if __name__ == "__main__":
    gen_list, disc_list = train()
    plot_res(gen_list = gen_list, disc_list=disc_list)
