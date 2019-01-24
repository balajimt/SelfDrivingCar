__author__ = "Balaji Muthazhagan, Anirudh GJ"

import scipy.misc
import random
from Configurations import xColumnDataset, yColumnDataset, noOfImages
from Configurations import trainBatchPointer, validationBatchPointer

# Training Dataset
trainX = xColumnDataset[:int(len(xColumnDataset) * 0.8)]
trainY = yColumnDataset[:int(len(xColumnDataset) * 0.8)]
noOfTraining = len(trainX)

# Validation Dataset
validationX = xColumnDataset[-int(len(xColumnDataset) * 0.2):]
validationY = yColumnDataset[-int(len(xColumnDataset) * 0.2):]
noOfValidation = len(validationX)

# Module to load return training data
def loadTrainData(batchSize):
    global trainBatchPointer
    xImage = []
    yRadians = []
    for counterVariable in range(0, batchSize):
        xImage.append(scipy.misc.imresize(scipy.misc.imread(trainX[(trainBatchPointer + counterVariable) % noOfTraining])[-150:], [66, 200]) / 255.0)
        yRadians.append([trainY[(trainBatchPointer + counterVariable) % noOfTraining]])
    trainBatchPointer += batchSize
    return xImage, yRadians

# Module to load return validation data
def loadValidationData(batchSize):
    global validationBatchPointer
    xImage = []
    yRadians = []
    for counterVariable in range(0, batchSize):
        xImage.append(scipy.misc.imresize(scipy.misc.imread(validationX[(validationBatchPointer + counterVariable) % noOfValidation])[-150:], [66, 200]) / 255.0)
        yRadians.append([validationY[(validationBatchPointer + counterVariable) % noOfValidation]])
    validationBatchPointer += batchSize
    return xImage, yRadians
