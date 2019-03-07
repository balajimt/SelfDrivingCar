__author__ = "Balaji Muthazhagan, Anirudh GJ"

import cv2
import math

# Steering Wheel Image and configurations
steeringImage = cv2.imread('img//SteeringWheel.jpg',0)
swImageRows,swImageCols = steeringImage.shape
steeringAngle = 0

# Data set parser information
xColumnDataset = []
yColumnDataset = []
with open("DrivingDatasetOutput//data.txt") as f:
    for line in f:
        try:
            xColumnDataset.append("DrivingDatasetOutput//" + line.split()[0])
            yColumnDataset.append(float(line.split()[1]) * math.pi / 180)
        except:
            pass
noOfImages = len(xColumnDataset)

# Counter
counterVariable = math.ceil(noOfImages*0.8)

# Norm constant for training
normConstant = 0.001

# Model save point
modelSavePoint = './save'

# Training Configs
noOfEpochs = 30
batchSize = 100

# Pointers to keep track of batch information
trainBatchPointer = 0
validationBatchPointer = 0
