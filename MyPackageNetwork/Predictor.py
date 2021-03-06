import torch

import MyPackageCommon.Constants as cst
import MyPackageNetwork.AdminiData as AD
import MyPackageNetwork.CreateDataLoader
import MyPackageNetwork.Learner


class Predictor:
    def __init__(self):
        self.learner = MyPackageNetwork.Learner.Learner()
        self.predictTargetDataLoader = None

    def setPredictTarget(self):
        # self.trainDataLoader,self.predictTargetDataLoader=GDL.getDataLoaderLatest()
        AD.setupPredictData()
        self.predictTargetDataLoader = AD.getPredictDataLoader()

    def predictFromDataLoader(self, place: int):
        if self.predictTargetDataLoader is None:
            self.setPredictTarget()
        for num, (X, y) in enumerate(self.predictTargetDataLoader):
            if num != place:
                if num > place:
                    return
                continue
            X, y = X.to(self.learner.device), y.to(self.learner.device)
            print('\n--------test data--------')
            print(X[0, 0])
            print("\n--------test data label--------")
            label = self.predictTargetDataLoader.dataset.classes[y.item()]
            print(label)

            print('--------all labels--------')
            print(self.predictTargetDataLoader.dataset.classes)

            pred = self.learner.model(X)
            print("--------predicted data--------")
            print(pred)
            print("--------maximum value--------")
            print('place:', pred.argmax(1).item() + 1)

    def predictTargetOrNot(self):
        if self.predictTargetDataLoader is None:
            self.setPredictTarget()

        print('--------all labels--------')
        classes = self.predictTargetDataLoader.dataset.classes
        print(classes)
        if not classes == [cst.labels.predictTarget]:
            print("データが対応してない")
            print("プログラム終了")
            exit()

        for num, (X, y) in enumerate(self.predictTargetDataLoader):
            X, y = X.to(self.learner.device), y.to(self.learner.device)
            # print(X)
            pred = self.learner.model(X)

            fileName=self.predictTargetDataLoader.dataset.fileNameList[num]
            probability=float(pred[0][0])
            correct=(pred.argmax(1) == y).type(torch.bool).sum().item()

            print(
                str(fileName) +
                "\t:\t" +
                str(correct)+
                "\t:\t" +
                str("{:10.4f}".format(float(pred[0][0])))+
                "," +
                str("{:10.4f}".format(float(pred[0][1])))
            )
            # print(pred)
            # print(pred.argmax(1))

    # todo not created
    def predictFromImage(self):
        # todo image from folder
        # todo image to dataLoader
        data = []
        for image in data:
            pred = self.learner.model(image)
            print(pred)



if __name__ == '__main__':
    predictor = Predictor()
    predictor.setPredictTarget()
    predictor.predictTargetOrNot()
