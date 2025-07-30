import torch 
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from DCGAN import DCgen,DCdisc
from torch import optim as opt
from tqdm.auto import tqdm 
from torchvision.utils import make_grid
import numpy as np
import random
import torch.nn.functional as F 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
#超参数
batch_size = 256
noise_dim = 64
lr = 0.0001
epochs = 10
beta_1 = 0.5
beta_2 = 0.999
device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()
n_classes = 10
mnist_shape = (1,28,28)

def show_tensor_images(image_tensor, epoch, num_images=25, size=(1, 28, 28)):
    '''
    可视化图像，确保只显示图像部分
    '''
    # 如果是拼接的tensor，分离出图像部分
    if image_tensor.shape[1] > size[0]:
        image_tensor = image_tensor[:, :size[0]]
        
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"./runs/res/DC-res-{epoch}")
    plt.show()


def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


def get_noise(num:int, noise_dim:int, device = device):
    noise = torch.randn(size=(num, noise_dim)).to(device)
    return noise
def combine_vector(x, y):
    return torch.concat((x.float(),y.float()),dim=1)

def train():
    dataloader = DataLoader(MNIST('./data',train=True,
                                  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), download=True),batch_size=batch_size,shuffle=True)
    gen_model = DCgen(noise_dim=noise_dim + n_classes).to(device)# G model
    gen_opt = opt.Adam(params=gen_model.parameters(),lr=lr, betas=(beta_1,beta_2))
    disc_model = DCdisc(image_channel=mnist_shape[0] + n_classes).to(device)# D model
    disc_opt = opt.Adam(params=disc_model.parameters(),lr=lr, betas=(beta_1,beta_2))
    gen_model.apply(weights_init)
    disc_model.apply(weights_init)
    gen_loss_list = []
    disc_loss_list = []
    for epoch in range(epochs):
        print(f"Epochs: {epoch + 1} / {epochs}")
        mean_disc_loss = 0
        mean_gen_loss = 0
        cur_len = 0
        fake_imgs = 0
        for real, label in tqdm(dataloader):
            cur_batch_size = len(real)
            #real = real.view(cur_batch_size, -1).to(device) #数据送上gpu
            real = real.to(device)
            label = label.to(device)
            one_hot_labels = F.one_hot(label, n_classes)
            imge_one_hot = one_hot_labels[:,:,None,None]
            imge_one_hot = imge_one_hot.repeat(1,1,mnist_shape[1],mnist_shape[2])
            fake_noise = get_noise(cur_batch_size, noise_dim=noise_dim, device=device)
            
            fake_noise_label = combine_vector(fake_noise,one_hot_labels)
            fake = gen_model(fake_noise_label)
            fake_imgs = fake
            fake = combine_vector(fake, imge_one_hot)
            real = combine_vector(real, imge_one_hot)
            fake_pred = disc_model(fake.detach())
            real_pred = disc_model(real)
            #更新disc
            disc_opt.zero_grad()
            #disc_loss = get_disc_loss(gen_model, disc_model, cur_batch_size, noise_dim=noise_dim,criterion=criterion, device=device, real=real)
    
            disc_loss = (criterion(fake_pred,torch.zeros_like(fake_pred)) + criterion(real_pred, torch.ones_like(real_pred))) / 2
            disc_loss.backward()
            disc_opt.step()
            #更新gen
            fake_pred = disc_model(fake)
            gen_opt.zero_grad()
            #gen_loss = get_gen_loss(gen=gen_model, disc=disc_model, batch_size=cur_batch_size, noise_dim=noise_dim, criterion=criterion, device=device)
            gen_loss = criterion(fake_pred, torch.ones_like(fake_pred))
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
                show_tensor_images(real,epoch=666)
                show_tensor_images(fake_imgs,epoch=epoch+1)  # 调用可视化函数
    torch.save(gen_model.state_dict(), f'./runs/model_params/Conditonal_DCgen_model_params-{epochs}.pth')
    # 保存判别器模型参数
    torch.save(disc_model.state_dict(), f'./runs/model_params/Conditonal_DCdisc_model_params-{epochs}.pth')
    return gen_loss_list, disc_loss_list
            
def plot_res(gen_list:list, disc_list:list):
    plt.figure(figsize=(10, 6))
    plt.plot(gen_list, label='Generator Loss')
    plt.plot(disc_list, label='Discriminator Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Conditonal_DCGAN Losses')
    plt.savefig(f"./runs/Conditonal_DCGAN_Losses-{epochs}")
    plt.show()

if __name__ == "__main__":
    gen_list, disc_list = train()
    plot_res(gen_list = gen_list, disc_list=disc_list)
