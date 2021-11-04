import MyPackageNetwork.CreateEditedImage as CEI

cld = CEI.CreateEditedData()

import MyPackageNetwork.NetWork as NW

cld.editByTargetList(NW.NNParams.targetLabels)

import MyPackageNetwork.CreateDataLoader as CLD

trainDataLoader,validDataLoader=CLD.getDataLoaderForTrain(NW.NNParams.targetLabels)

import pickle

import MyPackageCommon.Constants as Path

with open(Path.dataLoaderTrainLatest,mode="wb")as f:
    pickle.dump(trainDataLoader,f)
with open(Path.dataLoaderTestLatest,mode="wb")as f:
    pickle.dump(validDataLoader,f)