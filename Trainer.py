import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import DrivingData
import model

LOGDIR = './save'

tensorflowSession = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
tensorflowSession.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op =  tf.summary.merge_all()

savedWeightSession = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

noOfEpochs = 30
batchSize = 100

# train over the dataset about 30 times
for epoch in range(noOfEpochs):
  for counterVariable in range(int(DrivingData.noOfImages/batchSize)):
    xColumnDataset, yColumnDataset = DrivingData.loadTrainData(batchSize)
    train_step.run(feed_dict={model.x: xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 0.8})
    if counterVariable % 10 == 0:
      xColumnDataset, yColumnDataset = DrivingData.loadValidationData(batchSize)
      loss_value = loss.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batchSize + counterVariable, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xColumnDataset, model.y_: yColumnDataset, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * DrivingData.noOfImages/batchSize + counterVariable)

    if counterVariable % batchSize == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = savedWeightSession.save(tensorflowSession, checkpoint_path)
  print("Model saved in file: %s" % filename)

