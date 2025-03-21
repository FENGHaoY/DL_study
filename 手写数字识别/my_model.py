from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

path = './data'

trans = Compose([ToTensor()])

train = MNIST(path, train=True, download=True, transform=trans)
test = MNIST(path, train=False, download=True, transform=trans)

train_dl = DataLoader(train,batch_size=32,shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=True)

i, (inputs,targets) = next(enumerate(train_dl))

for i in range (32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(inputs[i][0],cmap='gray')

plt.show()