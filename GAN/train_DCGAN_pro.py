import torch 
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from DCGAN_pro import DCgen_model,DCdisc_model
from torch import optim as opt
from tqdm.auto import tqdm 
from torchvision.utils import make_grid
import numpy as np
import random

'''
超参数：
'''
noise_dim = 100
n_classes = 10
batch_size = 256
lr = 0.0002
epochs = 10
beta_1 = 0.5
beta_2 = 0.999
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda')
print(f"使用设备: {device}")  # 运行时会打印 "cuda" 或 "cpu"
save_path = "./runs/C_DCgan"
torch.manual_seed(40)
#data
def get_data():
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize(mean=[0.5], std=[0.5])])
    dataloader = DataLoader(MNIST('./data', train=True, transform=transform, download=True),batch_size=batch_size, shuffle=True)
    return dataloader

def get_noise(batch_size, noise_dim, device):
    return torch.randn(batch_size, noise_dim).to(device)

def get_one_hot(label, n_classes, device):
    return torch.nn.functional.one_hot(label, n_classes).float().to(device)

def show_generated_images(imgs, epoch):
    grid = make_grid(imgs[:25], nrow=5, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.savefig(f"{save_path}/epoch_{epoch}.png")
    plt.close()
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

fixed_noise = get_noise(25, noise_dim, device)
fixed_labels = torch.tensor([i for i in range(10)] * 2 + [0, 1, 2, 3, 4]).to(device)
fixed_labels_oh = get_one_hot(fixed_labels, n_classes, device)
#train model
def train():
    dataloader = get_data()
    G = DCgen_model(noise_dim=noise_dim, label_dim=n_classes, image_channel=1, f_map=64).to(device)
    D = DCdisc_model(label_dim=10, image_channel=1, f_map=64).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    G_opt = opt.Adam(params=G.parameters(), lr=lr, betas=(beta_1, beta_2))
    D_opt = opt.Adam(params=D.parameters(), lr=lr, betas=(beta_1, beta_2))
    G_loss = []
    D_loss = []
    for epoch in range(epochs):
        cur_D_loss = 0.0
        cur_G_loss = 0.0
        length = 0
        print(f"Epochs: {epoch + 1} / {epochs}")
        for real, label in tqdm(dataloader):
            real = real.to(device)
            label = label.to(device)
            cur_batch = len(real)
            label = get_one_hot(label, n_classes=n_classes, device=device)
            disc_real = D(real, label)
            real_loss = criterion(disc_real, torch.ones_like(disc_real))
            noise = get_noise(batch_size=cur_batch, noise_dim=noise_dim, device=device)
            fake_img = G(noise, label)
            disc_fake = D(fake_img.detach(), label)
            fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (real_loss + fake_loss) / 2 
            
            D_opt.zero_grad()
            disc_loss.backward()
            D_opt.step()
            
            disc_fake = D(fake_img, label)
            gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))
            
            G_opt.zero_grad()
            gen_loss.backward()
            G_opt.step()
            cur_D_loss += disc_loss.item()
            cur_G_loss += gen_loss.item()
            length += 1
        cur_D_loss = cur_D_loss / length
        cur_G_loss = cur_G_loss / length
        D_loss.append(cur_D_loss)
        G_loss.append(cur_G_loss)
        length = 0
        print(f"D_loss: {cur_D_loss}, G_loss: {cur_G_loss}")
        if (epoch+1) % 10 == 0 or epoch+1 == 1:
            G.eval()
            with torch.no_grad():
                fake_test = G(fixed_noise, fixed_labels_oh)
            show_generated_images(fake_test, epoch+1)
    torch.save(G.state_dict(), f"{save_path}/G_final.pth_{epochs}")
    torch.save(D.state_dict(), f"{save_path}/D_final.pth_{epochs}")
    return G_loss,D_loss

def plot_res(gen_list:list, disc_list:list):
    plt.figure(figsize=(10, 6))
    plt.plot(gen_list, label='Generator Loss')
    plt.plot(disc_list, label='Discriminator Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('C_DCGAN Losses')
    plt.savefig(f"./runs/C_DCGAN_Losses-{epochs}")
    plt.show()

if __name__ == "__main__":
    gen_list,disc_list = train()
    plot_res(gen_list, disc_list)