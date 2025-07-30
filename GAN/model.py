import torch.nn as nn
import torch
import torch.optim as optim
import torchsummary
class gen_block(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.gen_block = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):  # 添加 forward 方法
        return self.gen_block(x)
class gen(nn.Module):
    def __init__(self, noise_dim = 10, output_dim = 28*28, hidden_dim = 128):
        super().__init__()
        self.gen = nn.Sequential(
            gen_block(noise_dim, hidden_dim),
            gen_block(hidden_dim, 2 * hidden_dim),
            gen_block(2 * hidden_dim, 4 * hidden_dim),
            gen_block(4 * hidden_dim, 8 * hidden_dim),
            nn.Linear(8 * hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, noise):
        return self.gen(noise)
    
    
class disc_block(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.disc_block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):  # 添加 forward 方法
        return self.disc_block(x)

class disc(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dim = 128):
        super().__init__()
        self.disc = nn.Sequential(
            disc_block(input_dim, 4 * hidden_dim),
            disc_block(4 * hidden_dim, 2 * hidden_dim),
            disc_block(2 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,image):
        return self.disc(image)


if __name__ == "__main__":
    device = torch.device('cuda')
    gen = gen().to(device)
    disc = disc().to(device)
    torchsummary.summary(gen, input_size=(10,))
    torchsummary.summary(disc, input_size=(28*28,))