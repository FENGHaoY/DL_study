import torch.nn as nn
import torch
import torch.optim as optim
import torchsummary
import torchinfo
class DCgen_model(nn.Module):
    def __init__(self, noise_dim = 100, label_dim = 10, image_channel = 1, f_map = 64):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(noise_dim+label_dim, 7*7*f_map*2),
            nn.BatchNorm1d(7*7*f_map*2),
            nn.ReLU()
        )
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=f_map*2, out_channels=f_map, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f_map),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=f_map, out_channels=image_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    
    def forward(self,noise, label):
        x = torch.concat([noise, label], dim=1)
        x = self.pre(x)
        x = x.view(-1, 128, 7, 7)
        return self.gen(x)


class DCdisc_model(nn.Module):
    def __init__(self,label_dim = 10, image_channel = 1, f_map = 64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=label_dim + image_channel, out_channels=f_map, kernel_size=4, stride=2, padding=1), #14*14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=f_map, out_channels=f_map * 2, kernel_size=4, stride=2, padding=1),#7*7
            nn.BatchNorm2d(f_map * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f_map * 2, 1, 7)
        )
    
    def forward(self, image, label):
        label = label[:,:,None,None].repeat(1,1,28,28)
        image_label = torch.concat([image, label],dim=1)
        return self.disc(image_label).view(-1,1)

if __name__ == "__main__":
    device = torch.device("cuda")
    gen_model = DCgen_model().to(device)
    disc_model = DCdisc_model().to(device)
    torchinfo.summary(gen_model, 
            input_size=[(1, 100), (1, 10)],  # 指定两个输入的形状
            dtypes=[torch.float, torch.float],  # 指定输入数据类型
            device=device)
    torchinfo.summary(disc_model, 
            input_size=[(1, 1, 28, 28), (1, 10)],  # 图像和标签的形状
            dtypes=[torch.float, torch.float],
            device=device)