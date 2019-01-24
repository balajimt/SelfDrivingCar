__author__ = "Balaji Muthazhagan, Anirudh GJ"

import os
import tensorflow as tf

import model
import DrivingData
from Configurations import normConstant, modelSavePoint
from Configurations import noOfEpochs, batchSize

from tensorflow.core.protobuf import saver_pb2

# Create tensorflow session
tensorflowSession = tf.InteractiveSession()
trainableVariables = tf.trainable_variables()

# Define model loss and trainer
modelLoss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in trainableVariables]) * normConstant
modelTrainer = tf.train.AdamOptimizer(1e-4).minimize(modelLoss)
tensorflowSession.run(tf.initialize_all_variables())
tf.summary.scalar("Loss", modelLoss)
summary =  tf.summary.merge_all()

# Save weight session
savedWeightSession = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# Log into tensorboard
logs_path = './logs'
summaryFileWriter = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


# Train around the dataset
for epoch in range(noOfEpochs):
  for counterVariable in range(int(DrivingData.noOfImages/batchSize)):
    xColumnDataset, yColumnDataset = DrivingData.loadTrainData(batchSize)
    modelTrainer.run(feed_dict={model.x: xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 0.8})
    if counterVariable % 10 == 0:
      xColumnDataset, yColumnDataset = DrivingData.loadValidationData(batchSize)
      loss_value = modelLoss.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batchSize + counterVariable, loss_value))

	
    summary = summary.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
    summaryFileWriter.add_summary(summary, epoch * DrivingData.noOfImages/batchSize + counterVariable)

    if counterVariable % batchSize == 0:
      if not os.path.exists(modelSavePoint):
        os.makedirs(modelSavePoint)
      path = os.path.join(modelSavePoint, "model.ckpt")
      filename = savedWeightSession.save(tensorflowSession, path)
  print("Model successfully saved at: %s" % filename)

