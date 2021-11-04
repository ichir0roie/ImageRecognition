import os.path
import pickle
import shutil

import MyPackageCommon.Constants as cst
import MyPackageNetwork.CreateDataLoader as CLD
import MyPackageNetwork.CreateEditedImage as CEI

cei = CEI.CreateEditedData()


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


def savePack(packName: str):
    targetFolders = [
        cst.root.data,
        cst.root.images
    ]

    deletePack(packName)

    for folder in targetFolders:
        if not os.path.isdir(folder):
            continue
        savePath = cst.root.packs + packName + "/" + folder
        shutil.copytree(folder, savePath)


def deletePack(packName: str):
    packFolder = cst.root.packs + packName + "/"
    if os.path.isdir(packFolder):
        shutil.rmtree(packFolder)


def backUpPack():
    savePack("BACKUP")


def loadPack(packName: str):

    if not os.path.isdir(cst.root.packs+packName):
        raise

    backUpPack()
    targetFolders = [
        cst.root.data,
        cst.root.images
    ]
    for folder in targetFolders:
        if not os.path.isdir(folder):
            continue
        shutil.rmtree(folder)

    fromToList = [
        [
            cst.root.packs + packName + "/" + folder,
            folder
        ]
        for folder in targetFolders
    ]
    for fromFolder,toFolder in fromToList:
        shutil.copytree(fromFolder,toFolder)

def setupFolders():
    folders=[
        cst.root.data,
        cst.root.images,
        cst.root.packs,
        cst.data.edited,
        cst.data.dataLoader,
        cst.data.model,
        cst.images.learnTarget,
        cst.images.learnTargetNot,
        cst.images.predictTarget,
    ]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)

if __name__ == '__main__':
    # setupLearnData()

    # savePack("ABCD")
    # loadPack("ABCD")

    setupFolders()

