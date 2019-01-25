import tensorflow as tf
import scipy

def weightAssigner(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def biasAssigner(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

inputX = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = inputX

#first convolutional layer
weightConvolution1 = weightAssigner([5, 5, 3, 24])
biasConvolution1 = biasAssigner([24])

outputConvolution1 = tf.nn.relu(conv2d(x_image, weightConvolution1, 2) + biasConvolution1)

#second convolutional layer
weightConvolution2 = weightAssigner([5, 5, 24, 36])
biasConvolution2 = biasAssigner([36])

outputConvolution2 = tf.nn.relu(conv2d(outputConvolution1, weightConvolution2, 2) + biasConvolution2)

#third convolutional layer
weightConvolution3 = weightAssigner([5, 5, 36, 48])
biasConvolution3 = biasAssigner([48])

outputConvolution3 = tf.nn.relu(conv2d(outputConvolution2, weightConvolution3, 2) + biasConvolution3)

#fourth convolutional layer
weightConvolution4 = weightAssigner([3, 3, 48, 64])
biasConvolution4 = biasAssigner([64])

outputConvolution4 = tf.nn.relu(conv2d(outputConvolution3, weightConvolution4, 1) + biasConvolution4)

#fifth convolutional layer
weightConvolution5 = weightAssigner([3, 3, 64, 64])
biasConvolution5 = biasAssigner([64])

outputConvolution5 = tf.nn.relu(conv2d(outputConvolution4, weightConvolution5, 1) + biasConvolution5)

#FCL 1
weightFullyConnected1 = weightAssigner([1152, 1164])
biasFullyConnected1 = biasAssigner([1164])

outputConvolution5_flat = tf.reshape(outputConvolution5, [-1, 1152])
outputFullyConnected1 = tf.nn.relu(tf.matmul(outputConvolution5_flat, weightFullyConnected1) + biasFullyConnected1)

keep_prob = tf.placeholder(tf.float32)
outputFullyConnected1_drop = tf.nn.dropout(outputFullyConnected1, keep_prob)

#FCL 2
weightFullyConnected2 = weightAssigner([1164, 100])
biasFullyConnected2 = biasAssigner([100])

outputFullyConnected2 = tf.nn.relu(tf.matmul(outputFullyConnected1_drop, weightFullyConnected2) + biasFullyConnected2)

outputFullyConnected2_drop = tf.nn.dropout(outputFullyConnected2, keep_prob)

#FCL 3
weightFullyConnected3 = weightAssigner([100, 50])
biasFullyConnected3 = biasAssigner([50])

outputFullyConnected3 = tf.nn.relu(tf.matmul(outputFullyConnected2_drop, weightFullyConnected3) + biasFullyConnected3)

outputFullyConnected3_drop = tf.nn.dropout(outputFullyConnected3, keep_prob)

#FCL 3
weightFullyConnected4 = weightAssigner([50, 10])
biasFullyConnected4 = biasAssigner([10])

outputFullyConnected4 = tf.nn.relu(tf.matmul(outputFullyConnected3_drop, weightFullyConnected4) + biasFullyConnected4)

outputFullyConnected4_drop = tf.nn.dropout(outputFullyConnected4, keep_prob)

#Output
weightFullyConnected5 = weightAssigner([10, 1])
biasFullyConnected5 = biasAssigner([1])

y = tf.multiply(tf.atan(tf.matmul(outputFullyConnected4_drop, weightFullyConnected5) + biasFullyConnected5), 2) #scale the atan output
