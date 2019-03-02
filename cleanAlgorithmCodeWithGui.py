import numpy as np
import pandas as pd
from easygui import *

def prepareImage(directoryOfElement=None):
    img = Image.open(directoryOfElement)
    imgGray=img.convert('L')
    imgNP = np.asarray(imgGray)
    imgNp = imgNP/np.max(imgNP)
    imgT = tresholdEvo(imgNp,0.60,"both")
    return imgT

def tresholdEvo(img,t_value,tresholdType):
    if tresholdType == "both":
        img[np.where(img<t_value)] = 0
        img[np.where(img>t_value)] = 1
    elif tresholdType == "onlyb":
        mg[np.where(img<t_value)] = 0
    elif tresholdType == "onlys":
        img[np.where(img>t_value)] = 1
    return img

def readData(dirTrain,seperator=","):
    dirTrain = dirTrain.replace("/","\\")
    with open(dirTrain, 'r') as f:
        trainDirty = f.readlines()
    trainSplitted = []
    for elm in trainDirty:
        trainSplitted.append(elm.split(","))
    #trainSplittedNp = np.asarray(trainSplitted)
    return trainSplitted

def column(matrix, i):
    return [row[i] for row in matrix]

def normalizeStartingValue(dataset, columns):
    for cnt1 in columns:
        dataset[cnt1] = dataset[cnt1]-dataset[cnt1].min()
    return dataset

def bigNumEndIssue(dataset, maxValue, columns):
    for cnt1 in columns:
        print("GG",dataset)
        dataset[cnt1] = (dataset[cnt1]/dataset[cnt1].max())*maxValue
    return dataset

def categoricalDataIssue(dataset, columns):
    for cnt1 in columns:
        print("cat",cnt1)
        dataset[int(cnt1)] = dataset[int(cnt1)].astype("category").cat.codes
    return dataset

def createDataset(directoryOfDataset, sep, columnsCategorical, columnsNormalizeStart,columnsBigValues, maxValue):
    dataset = pd.read_csv(directoryOfDataset.replace("/","\\"),header=None,sep=sep)
    dataset1 = dataset
    """for elm in sorted(np.where(np.absolute(dataset1.corr().values[:,-1])<0.10), reverse=True):
        dataset = dataset.drop(columns = dataset.columns[elm])"""
    if not columnsCategorical == "":
        dataset = categoricalDataIssue(dataset, columnsCategorical)
    dataset = dataset.astype(float)
    if not columnsNormalizeStart == "":
        dataset = normalizeStartingValue(dataset, columnsNormalizeStart)
    if not columnsBigValues == "":
        dataset = bigNumEndIssue(dataset, maxValue, columnsBigValues)
    rawData = dataset.astype(int).values
    rowLength = int(np.nanmax(rawData))+1
    print("rowlength", int(np.max(rawData)))
    columnLength = len(rawData[0])
    realDataset = []
    for cnt1 in range(len(rawData)):
        zerosTemplate = np.zeros((rowLength,columnLength))
        for cnt2 in range(len(rawData[cnt1])):
            if not np.isnan(rawData[cnt1][cnt2]):
                zerosTemplate[int(rawData[cnt1][cnt2]),cnt2] = 1 #fill
        realDataset.append([zerosTemplate[:,0:-1],np.where(zerosTemplate[:,-1]==1)[0][0]])
    return realDataset,dataset1

def prepareDatasetForForging(dataSet,classStart,classLength,orderStart,orderEnd):
    indexClassList = []
    for cnt in range(classStart,classLength):
        indexClassList.append(np.where(np.asarray([row[-1] for row in dataSet]) == cnt)[0])
    dataSetClassesTogether = [[] for i in range(len(indexClassList))]
    for cnt1 in range(len(indexClassList)):
        for cnt2 in range(len(indexClassList[cnt1])):
            dataSetClassesTogether[cnt1].append(dataSet[int(indexClassList[cnt1][cnt2])][0])
    return dataSetClassesTogether

def comparisonCal(img1,img2):
    midTotal = np.sum(np.multiply(img1,img2))
    return midTotal/np.count_nonzero(img1)

def forgeProbabilityKernels(dataSet,classStart,classLength,orderStart,orderEnd,matchingNumber):
    dataMatrix1 = prepareDatasetForForging(dataSet,classStart,classLength,orderStart,orderEnd)
    dataMatrix2 = prepareDatasetForForging(dataSet,classStart,classLength,orderStart,orderEnd)
    NumberOfClasses = classLength
    posKerMat = [[] for i in range(NumberOfClasses)]
    while not np.asarray(dataMatrix1).size == 0:
        toBeAdded = [[] for i in range(NumberOfClasses)]
        forNextData = [[] for i in range(NumberOfClasses)]
        for cnt1 in range(NumberOfClasses):
            matchNumber = 0
            calculationMat = dataMatrix1[cnt1]
            for cnt2 in range(len(dataMatrix1[cnt1])):
                if comparisonCal(calculationMat[0],calculationMat[cnt2])>=matchingNumber:
                    toBeAdded[cnt1].append(cnt2)
                    matchNumber = matchNumber+1
                else:
                    forNextData[cnt1].append(cnt2)
        nextData = [[] for i in range(NumberOfClasses)]
        for cnt3 in range(NumberOfClasses):
            shapeForZeros = dataMatrix2[classStart][0].shape
            zerosMat = np.zeros(shapeForZeros)
            for element2 in np.unique(toBeAdded[cnt3]):
                zerosMat = zerosMat+dataMatrix1[cnt3][element2]
            posKerMat[cnt3].append(zerosMat)
        for cnt4 in range(NumberOfClasses):
            for element1 in forNextData[cnt4]:
                nextData[cnt4].append(dataMatrix1[cnt4][element1])
        dataMatrix1 = nextData
    posKerMatClean = [[] for i in range(NumberOfClasses)]
    cntP = 0
    for element1 in posKerMat:
        for element2 in element1:
            if not np.sum(element2) == 0.0:
                posKerMatClean[cntP].append(element2/np.max(element2))
        cntP = cntP+1
    return posKerMatClean, dataMatrix2

def renewPKfunc(posKerMat,yy):
    pkToRemove = []
    for cnt1 in range(len(yy)):
        for cnt2 in range(len(yy[cnt1])):
            if yy[cnt1][cnt2][1]!=yy[cnt1][cnt2][2]:
                pkToRemove.append([yy[cnt1][cnt2][1],yy[cnt1][cnt2][3]])
            else:
                break
    pkToRemoveSplitted = [[] for k in range(len(posKerMat))]
    cntOut = 0
    for cnt in [row[0] for row in pkToRemove]:
        pkToRemoveSplitted[cnt].append(pkToRemove[cntOut][1])
        cntOut = cntOut+1
    for cnt in range(len(pkToRemoveSplitted)):
        pkToRemoveSplitted[cnt] = sorted(np.unique(pkToRemoveSplitted[cnt]),reverse = True)
    #print(pkToRemoveSplitted)
    for cnt1 in range(len(pkToRemoveSplitted)):
        for cnt2 in range(len(pkToRemoveSplitted[cnt1])):
            #print(cnt1,cnt2,pkToRemoveSplitted[cnt1][cnt2])
            posKerMat[cnt1].pop(pkToRemoveSplitted[cnt1][cnt2])
    return posKerMat

def renewProbKernels(dataSet,orderStart,orderEnd,probabilityKernelMat):
    cnt1 = orderStart
    preds = []
    sortedYTotalNO = []
    dataToTest = dataSet[orderStart:orderEnd]
    actualClasses = [row[1] for row in dataToTest]
    while cnt1<orderEnd:
        #print("pred",cnt1)
        imgNP = dataSet[cnt1][0]
        yyyyy = [[] for i in range(orderEnd)]
        for cntElm1 in range(len(probabilityKernelMat)):
            cntCheck = 0
            for elm2 in probabilityKernelMat[cntElm1]:
                merging = np.multiply(elm2,imgNP)
                yyyyy[cnt1].append([np.sum(merging),cntElm1,int(actualClasses[cnt1]),cntCheck])
                cntCheck = cntCheck+1
        sortedY = sorted(yyyyy[cnt1],key=lambda column:column[0], reverse=True)
        preds.append(sortedY[0][1])
        if not int(actualClasses[cnt1]) == sortedY[0][1]:
            sortedYTotalNO.append(sortedY)
        cnt1 = cnt1+1
    probabilityKernelMat = renewPKfunc(probabilityKernelMat,sortedYTotalNO)
    return probabilityKernelMat

def predict(dataSet,orderStart,orderEnd,probabilityKernelMat):
    cnt1 = orderStart
    preds = []
    while cnt1<orderEnd:
        #print("pred",cnt1)
        imgNP = dataSet[cnt1][0]
        yyyyy = [[] for i in range(orderEnd)]
        cntCheck = 0
        for cntElm1 in range(len(probabilityKernelMat)):
            for elm2 in probabilityKernelMat[cntElm1]:
                merging = np.multiply(elm2,imgNP)
                yyyyy[cnt1].append([np.sum(merging),cntElm1])
            cntCheck = cntCheck+1
        sortedY = sorted(yyyyy[cnt1],key=lambda column:column[0], reverse=True)
        preds.append(sortedY[0][1])
        cnt1 = cnt1+1
    return preds

def testTheSystem(dataSet,orderStart,orderEnd, probabilityKernelMat):
    preds = predict(dataSet,orderStart,orderEnd,probabilityKernelMat)
    dataToTest = dataSet[orderStart:orderEnd]
    actualClasses = [row[1] for row in dataToTest]
    true = 0
    false = 0
    for cnt in range(len(actualClasses)):
        if preds[cnt] == actualClasses[cnt]:
            true = true+1
        else:
            false = false+1
    return true/(true+false)*100

def enterValues():
    msg = "Enter the Parameter Values"
    title = "Evo's Classifier"
    fieldNames = ["Directory", "Seperator", "Nan Value to Remove", "Categorical Columns", "Columns to Normalize to 0","Columns with Big Values", "Max Value for Big Value Columns","Start Value of Class","Number of Classes","Training Set Range","Testing Set Range","Matching Value"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = multenterbox(msg,title, fieldNames)
    
    # make sure that none of the fields was left blank
    while 1:
        if fieldValues == None: break
        errmsg = ""
        for i in [0,1,7,8,9,10,11]:
          if fieldValues[i].strip() == "":
            errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "": break # no problems found
        fieldValues = multenterbox(errmsg, title, fieldNames, fieldValues)
    return fieldValues

def stringToList(string):
    stringSplitted = string.split(",")
    realList = []
    for elm in stringSplitted:
        realList.append(int(elm))
    return realList

def prepareValues():
    directory,sep,nanValue,catColumns,normalizeToZeroColumns,bigValueColumns,maxValue,startValueOfClass,numOfClasses,trainingSetRange,testingSetRange,matchingValue = enterValues()
    if not catColumns == "":
        catColumns = stringToList(catColumns)
    if not normalizeToZeroColumns == "":
        normalizeToZeroColumns = stringToList(normalizeToZeroColumns)
    if not bigValueColumns == "":
        bigValueColumns = stringToList(bigValueColumns)
        maxValue = int(maxValue)
    
    startValueOfClass = int(startValueOfClass)
    numOfClasses = int(numOfClasses)
    
    trainingSetRange = trainingSetRange.split(",")
    trainingSetStart = int(trainingSetRange[0])
    trainingSetEnd = int(trainingSetRange[1])
    
    testingSetRange = testingSetRange.split(",")
    testingSetStart = int(testingSetRange[0])
    testingSetEnd = int(testingSetRange[1])
    
    matchingValue = float(matchingValue)
    
    return directory,sep,nanValue,catColumns,normalizeToZeroColumns,bigValueColumns,maxValue, startValueOfClass, numOfClasses, trainingSetStart, trainingSetEnd, testingSetStart, testingSetEnd, matchingValue

def saveTheProbabilityKernels(posKerMat):
    directory = enterbox("Enter Model Directory","Model Directory", "Directory")
    cnt = 0
    if directory[-1] == "/":
        corrector = "pk"
    else:
        corrector = "/pk"
    for elm in posKerMat:
        np.save("{}{}{}".format(directory,corrector,cnt),elm)
        cnt=cnt+1
        
def importPK():
    directory, rangeOfPK = multenterbox("Enter Model Directory","Model Directory", ["Directory","Number of Probability Kernels"])
    probabilityKernels = []
    if directory[-1] != "/":
        directory = directory+"/"
    directory = directory.replace("/","\\")
    for cnt in range(int(rangeOfPK)):
        probabilityKernels.append(np.load(directory+"pk{}.npy".format(cnt)))
    return probabilityKernels
    
#dirText, sep, nanValue, catColumns, normalizeToZeroColumns, bigValueColumns, maxValue, startValueOfClass, numOfClasses, trainingSetStart, trainingSetEnd, testingSetStart, testingSetEnd, matchingValue = prepareValues()
dirText, sep, nanValue, catColumns, normalizeToZeroColumns, bigValueColumns, maxValue, startValueOfClass, numOfClasses, trainingSetStart, trainingSetEnd, testingSetStart, testingSetEnd, matchingValue = "C:/Users/evrozm/Desktop/EvoClassifier/diabetes.csv", ",", "", "","",[1,2,3,4,5,6,7], 100, 0, 2, 267, 767, 0, 267, 0.5
dataSet,dataset1 = createDataset(dirText,sep,catColumns,normalizeToZeroColumns,bigValueColumns,maxValue)

choices = ["I don't have a model", "I have a model"]
choice = choicebox("Do you have a model?", "Model", choices)

if choice == "I don't have a model":
    posKerMat,dataMatrix = forgeProbabilityKernels(dataSet,startValueOfClass,numOfClasses,trainingSetStart,trainingSetEnd,matchingValue)
elif choice == "I have a model":
    posKerMat = importPK()

preds = predict(dataSet,testingSetStart,testingSetEnd,posKerMat)
accuracy = testTheSystem(dataSet,testingSetStart,testingSetEnd,posKerMat)
msgbox("Accuracy: {}, Predictions: {}".format(accuracy, preds))

msg = "Do you want to renew probability kernels?"
options = ["Yes","No"]
reply = buttonbox(msg, choices=options)
if reply == "Yes":
    posKerMat = renewProbKernels(dataSet,testingSetStart,testingSetEnd,posKerMat)
    print("PKs are renewed.")
elif reply == "No":
    print("PKs are not renewed.")

msg = "Do you want to save the model ?"
options = ["Yes","No"]
reply = buttonbox(msg, choices=options)
if reply == "Yes":
    saveTheProbabilityKernels(posKerMat)
    print("Model is saved.")
elif reply == "No":
    print("Model is not saved.")
