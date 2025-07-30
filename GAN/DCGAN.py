import torch.nn as nn
import torch
import torch.optim as optim
import torchsummary

def get_gen_block(in_channel, out_channel, kernal_size=3, stride=2, final=False):
    if not final:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernal_size,stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernal_size,stride=stride),
            nn.Tanh()
        )

class DCgen(nn.Module):
    def __init__(self, noise_dim = 10, out_channel = 1, hidden_dim = 64):
        super().__init__()
        self.noise_dim = noise_dim
        self.Dcgan = nn.Sequential(
            get_gen_block(noise_dim, hidden_dim * 4),
            get_gen_block(hidden_dim * 4 , hidden_dim * 2, kernal_size=4, stride=1),
            get_gen_block(hidden_dim * 2, hidden_dim),
            get_gen_block(hidden_dim, out_channel, kernal_size=4, final=True)
        )
        
    def forward(self, x):
        x = x.view(len(x), self.noise_dim, 1, 1)
        return self.Dcgan(x)

def get_disc_block(in_channel, out_channel, kernal_size=4, stride=2, final=False):
    if not final:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=kernal_size,stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernal_size, stride)
        )

class DCdisc(nn.Module):
    def __init__(self,image_channel = 1, hidden_dim = 16):
        super().__init__()
        self.DCdisc = nn.Sequential(
            get_disc_block(in_channel=image_channel, out_channel=hidden_dim),
            get_disc_block(hidden_dim, hidden_dim*2),
            get_disc_block(hidden_dim*2, 1,final=True),
        )
    
    def forward(self, image):
        disc_pred = self.DCdisc(image)
        return disc_pred.view(len(disc_pred), -1)

if __name__ == "__main__":
    device = torch.device("cuda")
    DCgen = DCgen().to(device)
    torchsummary.summary(DCgen,input_size=(10,))
    DCdisc = DCdisc().to(device)
    torchsummary.summary(DCdisc,input_size=(1,28,28))