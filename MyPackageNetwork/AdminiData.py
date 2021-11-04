
import MyPackageCommon.Constants as cst
import pickle


import MyPackageNetwork.CreateDataLoader as CLD
import MyPackageNetwork.CreateEditedImage as CEI
cei=CEI.CreateEditedData()

def setupLearnData():

    cei.createEditedLearnTargets()
    CLD.createLearnTargetDataLoader()

def setupPredictData():
    cei.createEditedPredictTarget()
    CLD.createPredictDataLoader()


def getLearnTargetDataLoader():
    with open(cst.dataLoader.latestTrain, mode="rb") as f:
        trainDataLoader = pickle.load(f)
    with open(cst.dataLoader.latestTest, mode="rb") as f:
        validDataLoader = pickle.load(f)

    return trainDataLoader, validDataLoader


def getPredictDataLoader():
    with open(cst.dataLoader.predict, mode="rb") as f:
        dataLoader = pickle.load(f)
    return dataLoader

if __name__ == '__main__':
    setupLearnData()