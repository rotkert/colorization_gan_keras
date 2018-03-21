import keras
import numpy as np
import dataset
import utils
import tensorflow as tf
from keras.models import load_model
from skimage import color
import time

def normalize_production(x):
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

model = load_model("vgg16_cifar10.h5")

_, mean = dataset.load_train_data(dataset = "cifar10", data_limit = 1000, colorspace = "YUV")
data, labels = dataset.load_valid_data(dataset = "cifar10", colorspace = "YUV", mean = mean)
data_y = data[:, :, :, :1]
data_y_noise = utils.add_noise(data_y)

model_gen = load_model("F:\\OneDrive - Politechnika Warszawska\\mgr-wyniki\\full_dataset\\C1_ganl1_model_transp_cifar10_YUV_bs128_run-2018-03-16_1522\\model_gen.h5")
model_gen.load_weights("C:\\Users\\Miko\\Desktop\\test\\weigths_epoch_425\\weights_gen.h5")
 
predicted_images = []
for i in range(data.shape[0]):
    y = data_y[i]
    y_noise = data_y_noise[i]
    uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
    yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
    yuv_pred[:, :, 0] += mean[0]
    yuv_pred[:, :, 1] += mean[1]
    yuv_pred[:, :, 2] += mean[2]
    predicted_images.append(yuv_pred)

data = np.array(predicted_images)
print(data[0,0,0, :])
data = color.yuv2rgb(data)
data *= 255
print(data[0,0,0, :])
data = normalize_production(data)
print(data[0,0,0, :])

y = keras.utils.to_categorical(labels, 10)

print(model.model.metrics_names)
ev = model.model.evaluate(data, y)
ev = np.round(np.array(ev), 4)
print(ev[0])
print(ev[1])
writer = tf.summary.FileWriter("C:\\Users\\Miko\\Desktop\\test")
summary = tf.Summary(value=[tf.Summary.Value(tag="aaaa", simple_value=ev[1]),])
writer.add_summary(summary, 425)
time.sleep(10)