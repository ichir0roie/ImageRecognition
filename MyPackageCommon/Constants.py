



class labels:
    learnTarget="LearnTarget"
    learnTargetNot="LearnTargetNot"
    predictTarget="PredictTarget"
    predicted="Predicted"

class data:
    data="Data/"
    original=data+"Original/"
    model=data+"Model/"
    edited=data+"Edited/"
    dataLoader=data+"DataLoader/"

class editedFolder:
    learnTarget=data.edited+labels.learnTarget+"/"
    learnTargetNot=data.edited+labels.learnTargetNot+"/"
    predictTarget = data.edited + labels.predictTarget + "/"
    predicted = data.edited + labels.predicted + "/"


class images:
    images="Images/"
    learnTarget=images+labels.learnTarget+"/"
    learnTargetNot=images+labels.learnTargetNot+"/"
    predictTarget=images+labels.predictTarget+"/"
    predicted=images+labels.predicted+"/"


    detectTarget=images+"DetectTarget"+"/"
    detectTargetAdjusted=images+"DetectTargetAdjusted"+"/"
    detectSample=images+"DetectSample"+"/"
    detectSampleAdjusted=images+"DetectSampleAdjusted"+"/"
    detectTemp=images+"DetectTemp"+"/"
    detected=images+"Detected"+"/"



class dataLoader:
    latestTrain= data.dataLoader + "train.pickle"
    latestTest= data.dataLoader + "test.pickle"
    predict= data.dataLoader + "predict.pickle"

class savePath:
    model= data.model + "learning.pth"
    structure= data.model + "structure.pickle"
    # latestWeightStr=data.model+"latestWeightStr"

