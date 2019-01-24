import cv2
import math

# Steering Wheel Image and configurations
steeringImage = cv2.imread('img//SteeringWheel.jpg',0)
swImageRows,swImageCols = steeringImage.shape
steeringAngle = 0

# Data set parser information
xColumnDataset = []
yColumnDataset = []
with open("DrivingDataset/data.txt") as f:
    for line in f:
        xColumnDataset.append("DrivingDataset/" + line.split()[0])
        yColumnDataset.append(float(line.split()[1]) * math.pi / 180)
noOfImages = len(xColumnDataset)