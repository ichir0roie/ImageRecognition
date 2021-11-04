import os
import pickle

import torch
import torch.nn.functional as F
from torch import nn

import MyPackageCommon.Constants as cst


class NNParams:
    targetLabels = [
        cst.labels.learnTarget,
        cst.labels.learnTargetNot
    ]

    resize = 100
    colorConvertMode = "L"  # "RGB"
    imageDepth = 2  # by colorConvertMode
    imageSize = resize * resize * imageDepth

    targetLabelSize = len(targetLabels)

    learnRate = 0.05
    epochs = 1000

    batchSize = 1

    targetAccuracy = 0.99


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)  # param 1 is need "1".
        self.pool = nn.MaxPool2d(3, 3)
        self.shapeConv1 = None
        self.conv2 = nn.Conv2d(3, 3, 3)

        self.loadStructure()

    def loadStructure(self):
        if os.path.exists(cst.savePath.structure):
            with open(cst.savePath.structure, "rb") as f:
                modelElems = pickle.load(f)
                self.fc1 = modelElems[0]
                self.fc2 = modelElems[1]
            self.setup = True
        else:
            self.setup = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        if not self.setup:
            self.fc1 = nn.Linear(x.shape[1], 100)
            self.fc2 = nn.Linear(100, 2)
            with open(cst.savePath.structure, "wb") as f:
                modelElems = [self.fc1, self.fc2]
                pickle.dump(modelElems, f)
            print("network auto setup")
            print(self)
            self.setup = True

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NNParams.imageSize, 10),
            nn.Sigmoid(),
            nn.Linear(10, NNParams.targetLabelSize),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CnnSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = CNN()
