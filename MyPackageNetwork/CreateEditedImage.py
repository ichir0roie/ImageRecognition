from PIL import Image
import glob
import os
import MyPackageCommon.Constants as cst
import MyPackageNetwork.NetWork as NW

class CreateEditedData:
    def __init__(self):
        self.imageDataList=[]

    def readPaths(self,readFolder):
        if readFolder[-1]!="/":
            readFolder=readFolder+"/"
        files=glob.glob(readFolder+"*")
        self.imageDataList=[]
        for i in files:
            imageData=ImageData()
            imageData.setPath(i)
            self.imageDataList.append(imageData)

    def deleteBeforeImages(self,readFolder):
        if readFolder[-1]!="/":
            readFolder=readFolder+"/"
        files = glob.glob(readFolder + "*")
        for file in files:
            os.remove(file)

    def resize(self):
        for imageData in self.imageDataList:
            img=self.getImageAndConvert(imageData.path)
            imageData.img=img

    def getImageAndConvert(self,path):
        img = Image.open(path)
        img=img.resize((NW.NNParams.resize,NW.NNParams.resize))
        img=img.convert(NW.NNParams.colorConvertMode)
        return img

    def saveImages(self,saveFolder,fileKeyWord):
        count=0
        for imageData in self.imageDataList:
            imageData.img.save(saveFolder+"/"+fileKeyWord+"_{}.png".format(count),format="png")
            count+=1

    def saveImagesForPredict(self,editedFolder):
        for imageData in self.imageDataList:
            imageData.img.save(editedFolder+imageData.fileName+".png",format("png"))

    def createEditedImagesByClassName(self, className):#className=FolderName in Original
        self.createEditedImages(
            originalFolder=cst.data.original + className,
            editedFolder=cst.data.edited + className,
            label=className
        )

        # os.removedirs(cst.data.edited+word)

    def createEditedLearnTargets(self):
        self.createEditedImages(
            originalFolder=cst.images.learnTarget,
            editedFolder=cst.editedFolder.learnTarget,
            label=cst.labels.learnTarget
        )
        self.createEditedImages(
            originalFolder=cst.images.learnTargetNot,
            editedFolder=cst.editedFolder.learnTargetNot,
            label=cst.labels.learnTargetNot
        )

    def createEditedPredictTarget(self):
        if not os.path.exists(cst.editedFolder.predictTarget):
            os.makedirs(cst.editedFolder.predictTarget)

        self.deleteBeforeImages(cst.editedFolder.predictTarget)

        self.readPaths(cst.images.predictTarget)
        self.resize()
        self.saveImagesForPredict(cst.editedFolder.predictTarget)


    def createEditedImages(self,originalFolder,editedFolder,label):
        if not os.path.exists(editedFolder):
            os.makedirs(editedFolder)

        self.deleteBeforeImages(editedFolder)

        self.readPaths(originalFolder)
        self.resize()
        self.saveImages(editedFolder, label)

    def editByTargetList(self,list):
        for i in list:
            self.createEditedImagesByClassName(i)

class ImageData:
    def __init__(self):
        self.path=None
        self.img=None
        self.fileName=None
        self.label=None

    def setPath(self,path:str):
        path=path.replace("\\","/")
        self.path=path
        fileName=path.split("/")[-1].split(".")[0]
        self.fileName=fileName


if __name__ == '__main__':
    cld=CreateEditedData()

    # cld.createEditedLearnTargets()
    cld.createEditedPredictTarget()