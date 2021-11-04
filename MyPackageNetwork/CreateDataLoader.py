# ライブラリの読み込み
from PIL import Image

import torch.utils.data as TUData
import torchvision

import glob
import pickle

import MyPackageNetwork.NetWork as network
import MyPackageCommon.Constants as cst


class DataSet(TUData.Dataset):

    def __init__(self, file_list, classes, phase='train'):
        self.file_list = file_list
        # self.transform = transform
        self.classes = classes
        self.phase = phase

        self.fileNameList=[self.getFilename(file) for file in file_list]

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

    def getFilename(self,fileName):
        fileName=fileName.replace("\\","/")
        fileName = fileName.split('/')[-1].split(".")[0]
        return fileName

class DataSetForPredict(DataSet):

    def __init__(self, file_list, classes, phase='train'):
        super().__init__(file_list, classes, phase)


    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)


        # ラベル名を数値に変換
        label = 0
        tensor = torchvision.transforms.functional.to_tensor(img)
        return tensor, label



def getDataLoaderForTrain():
    imagePaths = []
    classes=[
        cst.labels.learnTarget,
        cst.labels.learnTargetNot
    ]

    for imageClass in classes:
        globed = glob.glob(cst.data.edited + imageClass + "/*.png")
        imagePaths.extend(globed)

    # todo データ数が多くなってきたら、学習データを分割する必要がある？

    train_dataSet = DataSet(file_list=imagePaths, classes=classes, phase="train")
    valid_dataSet = DataSet(file_list=imagePaths, classes=classes, phase="valid")

    batchSize = network.NNParams.batchSize
    trainDataLoader = TUData.DataLoader(train_dataSet, batch_size=batchSize, shuffle=True)
    validDataLoader = TUData.DataLoader(valid_dataSet, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader


def getDataLoaderForPredict():
    imagesFolder = cst.editedFolder.predictTarget + "/"
    globed = glob.glob(imagesFolder + "*.png")
    classes=[
        cst.labels.predictTarget
    ]

    dataSet = DataSetForPredict(file_list=globed, classes=classes, phase="predict")
    dataLoader = TUData.DataLoader(dataSet, batch_size=1, shuffle=False)
    return dataLoader


def createLearnTargetDataLoader():
    t, v = getDataLoaderForTrain()
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
    dli = getDataLoaderForTrain()
    print(dli)
