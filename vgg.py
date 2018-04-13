from keras.models import load_model

import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from utils_evaluation import evaluator
from dataset import load_train_data, load_valid_data
from utils import process_after_predicted
from utils_visualisation import show_yuv
from skimage import color
import dataset

DATASET = "stl10"

evaluator = evaluator(dataset = DATASET)
data_yuv, mean = load_train_data(dataset = DATASET, data_limit = 5000, colorspace = "YUV")

# data, labels = dataset.load_stl10_train_data()
# mean = np.mean(data,axis=(0,1,2,3))
# std = np.std(data, axis=(0, 1, 2, 3))
# print(mean)
# print(std)
# a = data[0, 0, 0, ]
# print(a)

data_valid_yuv, lables_valid = load_valid_data(dataset = DATASET, colorspace = "YUV", mean = (0,0,0), size = 500)


data_valid = process_after_predicted(data_valid_yuv[:, :, :, 1:], data_valid_yuv[:, :, :, :1], (0,0,0), "YUV")

data_valid = color.rgb2gray(data_valid)
data_valid = color.gray2rgb(data_valid)


# for i in range(100):
#      show_yuv(data_valid[i], data_valid[i], (0, 0, 0))

print(evaluator.evaluate(data_valid, lables_valid))