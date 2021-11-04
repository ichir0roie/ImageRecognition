import MyPackageRectangleDetect.RectangleDetection as RD

import MyPackageCommon.Constants as cst

class DetectionManager:
    def __init__(self):

        self.rd=RD.RectangleDetection()
        self.targetPathList=[]

    def setupRD(self):
        self.rd=RD.RectangleDetection(setupMode=True)

    def initializeImages(self):
        # todo : delete before images
        return

    def detectImages(self):
        for path in self.targetPathList:
            self.detectImage(path)


    def detectImage(self,imagePath:str):
        correct=self.rd.detect(imagePath)

        print(
            "target:"+imagePath+","
            "result:"+str(correct)
        )



if __name__ == '__main__':
    dm=DetectionManager()
    dm.setupRD()
    # dm.detectImage(cst.images.detectTarget+"sample_1.png")
    dm.detectImage(cst.images.detectTarget+"T1.jpg")

