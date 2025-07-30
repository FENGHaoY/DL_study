import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
class CVAE(nn.Module):
    def __init__(self, label_dim, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim *2 , hidden_dim),
            nn.Tanh()
        )
        self.f1 = nn.Linear(hidden_dim, latent_dim)#mean
        self.f2 = nn.Linear(hidden_dim, latent_dim)#log(std)^2
        self.decode = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameter(self, mean, log_var):#重参数化，一种数学方式 我不懂。。
        noise = torch.randn_like(mean)
        z = mean + torch.exp(log_var*0.5)*noise
        return z
    
    def forward(self, x, label):
        x = torch.concat((x, label), dim=1)
        x = self.encode(x)
        mean = self.f1(x)
        log_var = self.f2(x)
        z = self.reparameter(mean, log_var)
        z = torch.concat((z, label), dim=1)
        res = self.decode(z)
        return res, mean, log_var
    
    def gen(self, znoise, label):
        znoise = torch.concat((znoise, label), dim=1)
        return self.decode(znoise)
    

if __name__ == "__main__":
    device = torch.device('cuda')
    model = CVAE(label_dim=10, input_dim=28*28, hidden_dim=256, latent_dim=128).to(device)
    torchsummary.summary(model=model, input_size=[(784,), (10,)])
        