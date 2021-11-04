import glob
import math

import cv2
import numpy as np

import MyPackageCommon.Constants as cst
from MyPackageRectangleDetect.DetectionSettings import *


class RectangleDetection:
    def __init__(self, setupMode=False):

        if setupMode:
            self.createDetectionParam()

        self.detectionSample = self.loadDetectionParam()
        if self.detectionSample is None:
            print("not found detection sample image.")
            print("try setup DetectionManager.")
            raise

        return

    def loadDetectionParam(self):
        image = cv2.imread(cst.images.detectSampleAdjusted + "detectSample.png", cv2.IMREAD_COLOR)
        return image

    def saveDetectionParam(self, image):
        cv2.imwrite(cst.images.detectSampleAdjusted + "detectSample.png", image)
        return

    def createDetectionParam(self):
        samplePaths = glob.glob(cst.images.detectSample + "*")
        images = []
        for samplePath in samplePaths:
            image = cv2.imread(samplePath, cv2.IMREAD_COLOR)
            image = self.adjustImage(image)
            images.append(image)

        images = np.array(images)
        assert images.ndim == 4, "すべての画像の大きさは同じでないといけない"

        mean_img = images.mean(axis=0)

        cv2.imwrite(cst.images.detectTemp+"sampleMean.png",mean_img)
        self.saveDetectionParam(mean_img)

        return

    def detect(self, targetImagePath: str):
        image, fileName = self.getImage(imagePath=targetImagePath)
        image = self.adjustImage(image)
        cv2.imwrite(cst.images.detectTemp + "adjusted.png", image)
        correct = self.detectImage(image)
        self.saveImage(image, fileName, correct, cst.images.detected)
        return correct

    def getImage(self, imagePath: str):
        fileName = imagePath.split("/")[-1].split(".")[0]
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        # height, width, channels = image.shape[:3]

        return image, fileName

    def adjustImage(self, image):
        if image is None:
            print("not set target image.")
            raise

        foundApprox = self.findSquares(image)

        # setImagePolyLines()
        imageCut = self.rectCut(image, foundApprox)

        return imageCut

        # pt0-> pt1およびpt0-> pt2からの

    def detectImage(self, image) -> bool:

        diff = self.detectionSample.astype(int) - image.astype(int)
        cv2.imwrite(cst.images.detectTemp + "diffs.png", diff)
        diffAbs = np.abs(diff)
        diffSum = diffAbs.sum()

        if diffSum < differenceThreshold:
            return True

        return False

    def saveImage(self, image, fileName: str, correct: bool, filePath: str):

        if image is None:
            return
        if correct:
            fileName = fileName + "_TRUE"
        else:
            fileName = fileName + "_FALSE"

        cv2.imwrite(filePath + fileName + ".png", image)

        return

    # ベクトル間の角度の余弦(コサイン)を算出
    def angle(self, pt1, pt2, pt0) -> float:
        dx1 = float(pt1[0, 0] - pt0[0, 0])
        dy1 = float(pt1[0, 1] - pt0[0, 1])
        dx2 = float(pt2[0, 0] - pt0[0, 0])
        dy2 = float(pt2[0, 1] - pt0[0, 1])
        v = math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))
        return (dx1 * dx2 + dy1 * dy2) / v

    # 画像上の四角形を検出
    def findSquares(self, image, cond_area=1000):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imwrite(cst.images.detectTemp+"binImage.png",bin_image)

        # 輪郭取得
        contours, _ = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        foundApprox = None

        for i, cnt in enumerate(contours):
            # 輪郭の周囲に比例する精度で輪郭を近似する
            arclen = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, arclen * 0.02, True)

            # 四角形の輪郭は、近似後に4つの頂点があります。
            # 比較的広い領域が凸状になります。

            # 凸性の確認 
            area = abs(cv2.contourArea(approx))
            if approx.shape[0] == 4 and area > cond_area and cv2.isContourConvex(approx):
                maxCosine = 0

                for j in range(2, 5):
                    # 辺間の角度の最大コサインを算出
                    cosine = abs(self.angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                    maxCosine = max(maxCosine, cosine)

                foundApprox = approx
                break

                # すべての角度の余弦定理が小さい場合
                # （すべての角度は約90度です）次に、quandrangeを書き込みます
                # 結果のシーケンスへの頂点
                # if maxCosine < 0.7 :
                # 四角判定!!
        if foundApprox is None:
            print("not found square.")
            raise

        return foundApprox

    def setImagePolyLines(self, image, foundApprox):
        if foundApprox is None:
            print("not found approx")
            raise

        rcnt = foundApprox.reshape(-1, 2)

        return cv2.polylines(image, [rcnt], True, (0, 0, 255), thickness=2, lineType=cv2.LINE_8)
        # print(rcnt)

    def rectCut(self, image, foundApprox):
        if foundApprox is None:
            print("image not found.")
            raise

        pts1 = np.float32(foundApprox)
        pts2 = np.float32(
            [
                [adjustImageWidth, adjustImageHeight],
                [adjustImageWidth, 0],
                [0, 0],
                [0, adjustImageHeight]
            ]
        )

        M = cv2.getPerspectiveTransform(pts1, pts2)

        imageCut = cv2.warpPerspective(
            image,
            M,
            (adjustImageWidth, adjustImageHeight))
        return imageCut


if __name__ == '__main__':
    rd = RectangleDetection(setupMode=True)
    # rd=RectangleDetection(setupMode=False)
