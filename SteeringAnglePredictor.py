__author__ = "Balaji Muthazhagan, Anirudh GJ"

import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
import math
import time


from Configurations import steeringImage,swImageRows,swImageCols,steeringAngle
from Configurations import xColumnDataset,yColumnDataset,noOfImages,counterVariable
import model

# Tensorflow session is created and the weight file is loaded
tensorflowSession = tf.InteractiveSession()
savedWeightSession = tf.train.Saver()
savedWeightSession.restore(tensorflowSession, "save/model.ckpt")

# Setting up time string for log creation
timestr = time.strftime("%Y%m%d-%H%M%S")

with open('logs//'+timestr+'.csv','a') as fd:
    fd.write("Predicted angle,Actual Angle\n"))

while(cv2.waitKey(10) != ord('q')):
    # Reading the image data
    colorImageData = scipy.misc.imread("DrivingDataset/" + str(counterVariable) + ".jpg", mode="RGB")
    grayscaleImageData = scipy.misc.imresize(colorImageData[-150:], [66, 200]) / 255.0
	
    # Predicted angle is got from the Model's predictions
    predictedAngle = model.y.eval(feed_dict={model.x: [grayscaleImageData], model.keep_prob: 1.0})[0][0] * 180.0 / math.pi
    
    # Writing to logs
    with open('logs//'+timestr+'.csv','a') as fd:
        fd.write(str(predictedAngle)+","+str(yColumnDataset[counterVariable]*180/math.pi)+"\n")
    
    # Showing the video frame and steering wheel on screen
    cv2.imshow("RoadView", cv2.cvtColor(colorImageData, cv2.COLOR_RGB2BGR))
	
    steeringAngle += 0.2 * pow(abs((predictedAngle - steeringAngle)), 2.0 / 3.0) * (predictedAngle - steeringAngle) / abs(predictedAngle - steeringAngle)
    steeringWheelImageRotation = cv2.getRotationMatrix2D((swImageCols/2,swImageRows/2),-steeringAngle,1)
    
    steeringWheelImage = cv2.warpAffine(steeringImage,steeringWheelImageRotation,(swImageCols,swImageRows))
    cv2.imshow("Steering Wheel Angle", steeringWheelImage)
    counterVariable += 1
	
cv2.destroyAllWindows()
