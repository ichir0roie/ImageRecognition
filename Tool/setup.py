
import os
import glob

folders = [
    "ImageTarget",
    "ImageTargetPredicted",
    "Data",

]

thisPlace=os.getcwd()
print(thisPlace)

thisCode=glob.glob("Tool/*")
print(thisCode)

if "Tool\\mainPredict.py" not in thisCode:
    print("作業ディレクトリが間違ってます。")
    print("ImageRecognitionを作業フォルダにしてください。")


for i in folders:
   if not os.path.exists(i):
       os.mkdir(i)



