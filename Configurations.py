__author__ = "Balaji Muthazhagan, Anirudh GJ"

import cv2
import math

# Steering Wheel Image and configurations
STEERING_IMAGE = cv2.imread('img//SteeringWheel.jpg', 0)
SW_IMAGE_ROWS, SW_IMAGE_COLS = STEERING_IMAGE.shape

# Norm constant for training
NORM_CONSTANT = 0.001

# Model save point
MODEL_SAVE_POINT = './save'

# Training Configs
NO_OF_EPOCHS = 30
BATCH_SIZE = 100

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
NO_OF_IMAGES = len(xColumnDataset)

# Counter
COUNTER_VARIABLE = math.ceil(NO_OF_IMAGES * 0.8)




# Pointers to keep track of batch information
trainBatchPointer = 0
validationBatchPointer = 0
