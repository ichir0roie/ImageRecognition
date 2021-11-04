# ライブラリの読み込み
from PIL import Image

import torch.utils.data as TUData
import torchvision

import glob
import pickle

import MyPackageNetwork.NetWork as network
import MyPackageCommon.Constants as cst


class Dataset(TUData.Dataset):

    def __init__(self, file_list, classes, phase='train'):
        self.file_list = file_list
        # self.transform = transform
        self.classes = classes
        self.phase = phase

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # 画像ラベルをファイル名から抜き出す
        label = self.file_list[index].replace("\\","/").split('/')[-2]

        # ラベル名を数値に変換
        label = self.classes.index(label)
        tensor = torchvision.transforms.functional.to_tensor(img)
        return tensor, label

class DataSetWithName(Dataset):

    def __init__(self, file_list, classes, phase='train'):
        super().__init__(file_list, classes, phase)
        self.fileNameList=[self.getFilename(file) for file in file_list]

    def getFilename(self,fileName):
        fileName=fileName.replace("\\","/")
        fileName = fileName.split('/')[-1].split(".")[0]
        return fileName


def getDataLoaderForTrain(imageClasses: list):
    imagePaths = []
    for imageClass in imageClasses:
        globed = glob.glob(cst.data.edited + imageClass + "/*.png")
        imagePaths.extend(globed)
    classes = imageClasses

    # todo データ数が多くなってきたら、学習データを分割する必要がある？

    train_dataSet = DataSetWithName(file_list=imagePaths, classes=classes, phase="train")
    valid_dataSet = DataSetWithName(file_list=imagePaths, classes=classes, phase="valid")

    batchSize = network.NNParams.batchSize
    trainDataLoader = TUData.DataLoader(train_dataSet, batch_size=batchSize, shuffle=True)
    validDataLoader = TUData.DataLoader(valid_dataSet, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader


def getDataLoaderForPredict():
    imagesFolder = cst.editedFolder.predictTarget + "/"
    globed = glob.glob(imagesFolder + "*.png")

    dataSet = DataSetWithName(file_list=globed, classes=[cst.labels.predictTarget], phase="predict")
    dataLoader = TUData.DataLoader(dataSet, batch_size=1, shuffle=False)
    return dataLoader


def createLearnTargetDataLoader():
    t, v = getDataLoaderForTrain(network.NNParams.targetLabels)
    with open(cst.dataLoader.latestTrain, mode="wb") as f:
        pickle.dump(t, f)
    with open(cst.dataLoader.latestTest, mode="wb") as f:
        pickle.dump(v, f)


def createPredictDataLoader():
    dl = getDataLoaderForPredict()
    with open(cst.dataLoader.predict, mode="wb") as f:
        pickle.dump(dl, f)




if __name__ == '__main__':
    # dli=getDataLoader(["dogs","cats"])
    dli = getDataLoaderForTrain(network.NNParams.targetLabels)
    print(dli)
