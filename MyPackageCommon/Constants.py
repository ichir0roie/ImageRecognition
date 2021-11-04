class labels:
    learnTarget = "LearnTarget"
    learnTargetNot = "LearnTargetNot"
    predictTarget = "PredictTarget"
    predicted = "Predicted"


class root:
    data = "Data/"
    images = "Images/"
    packs = "Packs/"


class data:
    original = root.data + "Original/"
    model = root.data + "Model/"
    edited = root.data + "Edited/"
    dataLoader = root.data + "DataLoader/"


class editedFolder:
    learnTarget = data.edited + labels.learnTarget + "/"
    learnTargetNot = data.edited + labels.learnTargetNot + "/"
    predictTarget = data.edited + labels.predictTarget + "/"
    predicted = data.edited + labels.predicted + "/"


class images:
    learnTarget = root.images + labels.learnTarget + "/"
    learnTargetNot = root.images + labels.learnTargetNot + "/"
    predictTarget = root.images + labels.predictTarget + "/"
    predicted = root.images + labels.predicted + "/"

    detectTarget = root.images + "DetectTarget" + "/"
    detectTargetAdjusted = root.images + "DetectTargetAdjusted" + "/"
    detectSample = root.images + "DetectSample" + "/"
    detectSampleAdjusted = root.images + "DetectSampleAdjusted" + "/"
    detectTemp = root.images + "DetectTemp" + "/"
    detected = root.images + "Detected" + "/"


class dataLoader:
    latestTrain = data.dataLoader + "train.pickle"
    latestTest = data.dataLoader + "test.pickle"
    predict = data.dataLoader + "predict.pickle"


class savePath:
    model = data.model + "learning.pth"
    structure = data.model + "structure.pickle"
    # latestWeightStr=data.model+"latestWeightStr"
