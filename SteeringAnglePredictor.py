__author__ = "Balaji Muthazhagan, Anirudh GJ"

import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
import math

from Configurations import steeringImage,swImageRows,swImageCols,steeringAngle
from Configurations import xColumnDataset,yColumnDataset,noOfImages
import model

# Tensorflow session is created and the weight file is loaded
tensorflowSession = tf.InteractiveSession()
savedWeightSession = tf.train.Saver()
savedWeightSession.restore(tensorflowSession, "save/model.ckpt")

counterVariable = math.ceil(noOfImages*0.8)
print("Starting frameofvideo:" +str(counterVariable))

while(cv2.waitKey(10) != ord('q')):
    colorImageData = scipy.misc.imread("DrivingDataset/" + str(counterVariable) + ".jpg", mode="RGB")
    grayscaleImageData = scipy.misc.imresize(colorImageData[-150:], [66, 200]) / 255.0
    predictedAngle = model.y.eval(feed_dict={model.x: [grayscaleImageData], model.keep_prob: 1.0})[0][0] * 180.0 / math.pi
    
    print("Steering angle: " + str(predictedAngle) + " (pred)\t" + str(yColumnDataset[counterVariable]*180/math.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(colorImageData, cv2.COLOR_RGB2BGR))
	
    steeringAngle += 0.2 * pow(abs((predictedAngle - steeringAngle)), 2.0 / 3.0) * (predictedAngle - steeringAngle) / abs(predictedAngle - steeringAngle)
    M = cv2.getRotationMatrix2D((swImageCols/2,swImageRows/2),-steeringAngle,1)
    steeringWheelImage = cv2.warpAffine(steeringImage,M,(swImageCols,swImageRows))
    cv2.imshow("Steering Wheel Angle", steeringWheelImage)
    counterVariable += 1

cv2.destroyAllWindows()
