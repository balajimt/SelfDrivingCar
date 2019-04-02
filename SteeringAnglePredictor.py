__author__ = "Balaji Muthazhagan, Anirudh GJ"

import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
import math
import time


from Configurations import STEERING_IMAGE,SW_IMAGE_ROWS,SW_IMAGE_COLS,steeringAngle
from Configurations import xColumnDataset,yColumnDataset,COUNTER_VARIABLE
import model

# Tensorflow session is created and the weight file is loaded
tensorflowSession = tf.InteractiveSession()
savedWeightSession = tf.train.Saver()
savedWeightSession.restore(tensorflowSession, "save/model.ckpt")

# Setting up time string for log creation
timestr = time.strftime("%Y%m%d-%H%M%S")

with open('logs//'+timestr+'.csv','a') as fd:
    fd.write("Predicted angle,Actual Angle\n")

while(cv2.waitKey(10) != ord('q')):
    COUNTER_VARIABLE += 1
    try:
        # Read image data
        colorImageData = scipy.misc.imread(xColumnDataset[COUNTER_VARIABLE], mode="RGB")
        grayscaleImageData = scipy.misc.imresize(colorImageData[-150:], [66, 200]) / 255.0
            
        # Predicted angle is got from the Model's predictions
        predictedAngle = model.y.eval(feed_dict={model.inputX: [grayscaleImageData], model.keep_prob: 1.0})[0][0] * 180.0 / math.pi
        
        # Writing to logs
        with open('logs//'+timestr+'.csv','a') as fd:
            fd.write(str(predictedAngle)+","+str(yColumnDataset[COUNTER_VARIABLE]*180/math.pi)+"\n")
        
        # Showing the video frame and steering wheel on screen
        cv2.imshow("RoadView", cv2.cvtColor(colorImageData, cv2.COLOR_RGB2BGR))
            
        steeringAngle += 0.2 * (predictedAngle - steeringAngle)
        steeringWheelImageRotation = cv2.getRotationMatrix2D((SW_IMAGE_COLS / 2, SW_IMAGE_ROWS / 2), -steeringAngle, 1)
        
        steeringWheelImage = cv2.warpAffine(STEERING_IMAGE, steeringWheelImageRotation, (SW_IMAGE_COLS, SW_IMAGE_ROWS))
        cv2.imshow("Steering Wheel Angle", steeringWheelImage)
    except:
        pass
cv2.destroyAllWindows()


