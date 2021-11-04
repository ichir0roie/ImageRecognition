import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download training Data from open datasets.
training_data = datasets.FashionMNIST(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor(),
)
with open("fashionMnistTrain.pickle", mode="wb")as f:
    pickle.dump(training_data, f)

# Download test Data from open datasets.
test_data = datasets.FashionMNIST(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor(),
)


with open("fashionMnistTest.pickle", mode="wb")as f:
    pickle.dump(test_data, f)
