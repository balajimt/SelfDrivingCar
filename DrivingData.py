import scipy.misc
import random

xColumnDataset = []
yColumnDataset = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("DrivingDataset/data.txt") as f:
    for line in f:
        xColumnDataset.append("DrivingDataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        yColumnDataset.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
noOfImages = len(xColumnDataset)


train_xs = xColumnDataset[:int(len(xColumnDataset) * 0.8)]
train_ys = yColumnDataset[:int(len(xColumnDataset) * 0.8)]

val_xs = xColumnDataset[-int(len(xColumnDataset) * 0.2):]
val_ys = yColumnDataset[-int(len(xColumnDataset) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def loadTrainData(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for counterVariable in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + counterVariable) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + counterVariable) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def loadValidationData(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for counterVariable in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + counterVariable) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + counterVariable) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
