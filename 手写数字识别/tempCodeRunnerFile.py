from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from pandas import read_csv
from matplotlib import pyplot as plt
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
from numpy import vstack
from numpy import argmax
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)