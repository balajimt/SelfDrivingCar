__author__ = "Balaji Muthazhagan, Anirudh GJ"

import os
import tensorflow as tf

import model
import DrivingData
from Configurations import NORM_CONSTANT, MODEL_SAVE_POINT
from Configurations import NO_OF_EPOCHS, BATCH_SIZE

from tensorflow.core.protobuf import saver_pb2

# Create tensorflow session
tensorflowSession = tf.InteractiveSession()
trainableVariables = tf.trainable_variables()

# Define model loss and trainer
modelLoss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in trainableVariables]) * NORM_CONSTANT
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
for epoch in range(NO_OF_EPOCHS):
  for counterVariable in range(int(DrivingData.NO_OF_IMAGES / BATCH_SIZE)):
    xColumnDataset, yColumnDataset = DrivingData.loadTrainData(BATCH_SIZE)
    modelTrainer.run(feed_dict={model.x: xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 0.8})
    if counterVariable % 10 == 0:
      xColumnDataset, yColumnDataset = DrivingData.loadValidationData(BATCH_SIZE)
      loss_value = modelLoss.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * BATCH_SIZE + counterVariable, loss_value))

	
    summary = summary.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
    summaryFileWriter.add_summary(summary, epoch * DrivingData.NO_OF_IMAGES / BATCH_SIZE + counterVariable)

    if counterVariable % BATCH_SIZE == 0:
      if not os.path.exists(MODEL_SAVE_POINT):
        os.makedirs(MODEL_SAVE_POINT)
      path = os.path.join(MODEL_SAVE_POINT, "model.ckpt")
      filename = savedWeightSession.save(tensorflowSession, path)
  print("Model successfully saved at: %s" % filename)

