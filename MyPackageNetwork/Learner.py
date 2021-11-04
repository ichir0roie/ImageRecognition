import glob
import os

import torch
from torch import nn

import MyPackageCommon.Constants as cst
import MyPackageNetwork.AdminiData
import MyPackageNetwork.NetWork as network


# import matplotlib.pyplot as plt

class Learner:
    def __init__(self, modeDeleteOldModel=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} self.device".format(self.device))

        self.trainDataLoader, self.validDataLoader = MyPackageNetwork.AdminiData.getLearnTargetDataLoader()

        for X, y in self.validDataLoader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype)
            break

        # self.model = network.FFNN().to(self.device)
        self.model = network.CNN().to(self.device)

        # print(self.model)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=network.NNParams.learnRate)

        self.modeDeleteOldModel = modeDeleteOldModel

        if os.path.exists(cst.savePath.model):
            try:
                self.model.load_state_dict(torch.load(cst.savePath.model, map_location=torch.device('cpu')))
                self.model.loadStructure()
            except Exception as e:
                self.model.setup = False
                if self.modeDeleteOldModel:
                    files = glob.glob(cst.data.model + "*")
                    for file in files:
                        os.remove(file)
                else:
                    raise "data model is changed. please escape before model data."

        self.model.eval()
        # print(self.model.conv1.weight)
        # print(self.model.conv2.weight)
        # print(self.model.fc1.weight)
        # print(self.model.fc2.weight)

    def train(self):
        size = len(self.trainDataLoader.dataset)
        for batch, (X, y) in enumerate(self.trainDataLoader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        self.model.eval()

    def viewTest(self):
        size = len(self.validDataLoader.dataset)
        num_batches = len(self.validDataLoader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.validDataLoader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} ")

        return correct, test_loss

    def run(self):

        print("testTargets : " + str(self.trainDataLoader.dataset.classes))

        epochs = network.NNParams.epochs

        # for t in range(epochs):
        times = 0
        beforeLoss = 0

        # for t in range(epochs):
        loop = True
        while loop:
            self.train()
            self.test()
            if times % (epochs / 10) == 0:
                print(f"Epoch {times + 1}\n-------------------------------")
                accuracy, loss = self.viewTest()
                lossReduction = beforeLoss - loss
                predictedTimesToZero = loss / lossReduction
                print("loss reduction : " + str(lossReduction))
                print("Predicted number of times to loss 0 : " + str(predictedTimesToZero))

                # if predictedTimesToZero > 10000:
                #     # print("maybe learning will not succeed.")
                #     print("can't learning more.")
                #     print("stop program.")
                #     break

                if accuracy >= network.NNParams.targetAccuracy:
                    print("maybe finish learning.")
                    print("stop program.")
                    break

                beforeLoss = loss
                torch.save(self.model.to('cpu').state_dict(), cst.savePath.model)
                # latestWeightStr=str(self.model.conv1.weight)
                # with open(cst.savePath.latestWeightStr,mode="w",encoding="utf-8")as f:
                #     f.write(latestWeightStr)

            times += 1

        print("Done!")


if __name__ == '__main__':
    learner = Learner(modeDeleteOldModel=True)
    learner.run()
